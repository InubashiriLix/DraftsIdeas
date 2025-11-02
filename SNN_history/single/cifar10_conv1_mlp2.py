import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import optuna

# SpikingJelly 的激活式 API
from spikingjelly.activation_based import layer, neuron, encoding, functional, surrogate

###############################################################################
# 硬编码超参
###############################################################################
EPOCHS = 10
N_TRIALS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # 如果在 Windows 或者资源有限的环境中，可能需要改小，例如 0。


###############################################################################
# 数据加载 (CIFAR-10)
###############################################################################
def get_cifar10_loaders(batch_size=64, num_workers=4):
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


###############################################################################
# 构建前面加一个卷积层的 SNN
###############################################################################
def build_mlp_with_conv_front(tau: float, hidden_size: int):
    net = nn.Sequential(
        layer.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        layer.Flatten(),
        layer.Linear(32 * 32 * 32, hidden_size, bias=False),
        neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        layer.Linear(hidden_size, 10, bias=False),
        neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
    )
    return net


###############################################################################
# 训练 & 测试
###############################################################################
def train_one_epoch(net, train_loader, encoder, optimizer, T, device):
    net.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_samples = 0

    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        label_onehot = F.one_hot(label, 10).float()

        optimizer.zero_grad()

        out_fr = 0.0
        for _ in range(T):
            spike_img = encoder(img)
            out_fr += net(spike_img)
        out_fr /= T

        loss = F.mse_loss(out_fr, label_onehot)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * label.numel()
        preds = out_fr.argmax(dim=1)
        train_correct += (preds == label).sum().item()
        train_samples += label.numel()

        functional.reset_net(net)

    avg_loss = train_loss_sum / train_samples
    accuracy = 100.0 * train_correct / train_samples
    return avg_loss, accuracy


def test_one_epoch(net, test_loader, encoder, T, device):
    net.eval()
    test_loss_sum = 0.0
    test_correct = 0
    test_samples = 0

    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, 10).float()

            out_fr = 0.0
            for _ in range(T):
                spike_img = encoder(img)
                out_fr += net(spike_img)
            out_fr /= T

            loss = F.mse_loss(out_fr, label_onehot)
            test_loss_sum += loss.item() * label.numel()
            preds = out_fr.argmax(dim=1)
            test_correct += (preds == label).sum().item()
            test_samples += label.numel()

            functional.reset_net(net)

    avg_loss = test_loss_sum / test_samples
    accuracy = 100.0 * test_correct / test_samples
    return avg_loss, accuracy


###############################################################################
# Optuna 搜索的 Objective
###############################################################################
def objective(trial: optuna.Trial):
    T = trial.suggest_int("T", 5, 20, step=5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    tau = trial.suggest_float("tau", 1.0, 4.0)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_size = trial.suggest_int("hidden_size", 64, 512, log=True)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, num_workers=NUM_WORKERS
    )

    net = build_mlp_with_conv_front(tau, hidden_size=hidden_size).to(device=DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    encoder = encoding.PoissonEncoder()

    best_acc = 0.0
    for ep in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            net, train_loader, encoder, optimizer, T, DEVICE
        )
        test_loss, test_acc = test_one_epoch(net, test_loader, encoder, T, DEVICE)

        if test_acc > best_acc:
            best_acc = test_acc

        print(
            f"[Trial #{trial.number}] Epoch {ep + 1}/{EPOCHS} | "
            f"T={T}, lr={lr:.4e}, tau={tau:.2f}, batch={batch_size}, hidden={hidden_size} | "
            f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%"
        )

        trial.report(test_acc, ep)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_acc


def run_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n===== Optuna 搜索结束 =====")
    print(f"Best trial idx: {study.best_trial.number}")
    print("Best hyperparams:", study.best_params)
    print(f"Best test_acc: {study.best_value:.2f}%")


def main():
    print(f"固定 EPOCHS={EPOCHS}, 硬编码 n_trials={N_TRIALS}, 设备={DEVICE}")
    run_optuna()


if __name__ == "__main__":
    main()
