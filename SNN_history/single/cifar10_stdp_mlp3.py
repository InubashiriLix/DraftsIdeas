import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import optuna  # 需要先 pip install optuna

# 如果以下 SpikingJelly 导入报错，请根据自己版本或安装位置调整
from spikingjelly.activation_based import neuron, surrogate, functional, base


###############################
# 1) 数据加载: CIFAR-10
###############################
def get_cifar10_loaders(batch_size=64, num_workers=2):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
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


###############################
# 2) Label-driven STDP Learner (仅最后一层用)
###############################
class LabelDrivenSTDPLearner(base.MemoryModule):
    """
    用于输出层(10维) 的“标签驱动 STDP”
    dw_ij = lr * sum_batch( pre_i * (label_j - post_j) )
    """

    def __init__(self, synapse: nn.Linear, sn: neuron.BaseNode, lr=1e-3):
        super().__init__()
        self.synapse = synapse
        self.sn = sn
        self.lr = lr

        self.register_memory("pre_spike", None)
        self.register_memory("post_spike", None)
        self.register_memory("label_spike", None)

    def record_pre_spike(self, pre_spike: torch.Tensor):
        self.pre_spike = pre_spike

    def record_post_spike(self, post_spike: torch.Tensor):
        self.post_spike = post_spike

    def record_label_spike(self, label_spike: torch.Tensor):
        self.label_spike = label_spike

    def run_step(self):
        if (
            self.pre_spike is None
            or self.post_spike is None
            or self.label_spike is None
        ):
            return
        diff = self.label_spike - self.post_spike  # (B, 10)
        dw = torch.einsum("bi,bj->ji", self.pre_spike, diff)
        with torch.no_grad():
            self.synapse.weight += self.lr * dw

        self.pre_spike = None
        self.post_spike = None
        self.label_spike = None

    def reset(self):
        self.pre_spike = None
        self.post_spike = None
        self.label_spike = None
        super().reset()


###############################
# 3) 更宽的多层网络:
#    2层隐藏层 + 最后一层做 label-driven STDP
###############################
class WiderLabelSTDPNet(nn.Module):
    """
    CIFAR-10 输入: (B,3,32,32) => Flatten => 2层 MLP => 最后一层 (10) => LIF => LabelDrivenSTDP
    具体结构:
      3072 -> h1 -> LIF1 -> h2 -> LIF2 -> 10 -> LIFout
    """

    def __init__(self, h1=512, h2=256, lr=1e-3, tau_lif=2.0):
        super().__init__()

        # 第1层: 3072->h1
        self.fc1 = nn.Linear(32 * 32 * 3, h1, bias=False)
        self.lif1 = neuron.LIFNode(tau=tau_lif, surrogate_function=surrogate.ATan())

        # 第2层: h1->h2
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.lif2 = neuron.LIFNode(tau=tau_lif, surrogate_function=surrogate.ATan())

        # 输出层: h2->10
        self.fc_out = nn.Linear(h2, 10, bias=False)
        self.lif_out = neuron.LIFNode(tau=tau_lif, surrogate_function=surrogate.ATan())

        # LabelDrivenSTDP 只在最后一层
        self.label_stdp = LabelDrivenSTDPLearner(self.fc_out, self.lif_out, lr=lr)

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        bsz = x.shape[0]
        x = x.view(bsz, -1)  # flatten

        # 第1层
        x1 = self.fc1(x)
        x1_lif = self.lif1(x1)

        # 第2层
        x2 = self.fc2(x1_lif)
        x2_lif = self.lif2(x2)

        # 输出层
        x_out = self.fc_out(x2_lif)
        pre_spike_out = (x2_lif > 0).float()
        self.label_stdp.record_pre_spike(pre_spike_out)

        out_lif = self.lif_out(x_out)
        post_spike_out = (out_lif > 0).float()
        self.label_stdp.record_post_spike(post_spike_out)

        # 如果有 label, 生成 label_spike
        if label is not None:
            label_spike = torch.zeros_like(post_spike_out)
            label_spike.scatter_(1, label.view(-1, 1), 1.0)
            self.label_stdp.record_label_spike(label_spike)

        return out_lif

    def run_step(self):
        self.label_stdp.run_step()

    def reset(self):
        self.lif1.reset()
        self.lif2.reset()
        self.lif_out.reset()
        self.label_stdp.reset()


###############################
# 4) Train & Test
###############################
def train_epoch(net, dataloader, device, time_steps=2):
    net.train()
    correct = 0
    total = 0
    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)

        net.reset()
        out_fr = 0.0
        for t in range(time_steps):
            out_t = net(img, label=label)  # 训练时提供 label
            net.run_step()  # STDP update
            out_fr += out_t

        pred = out_fr.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.numel()

    return 100.0 * correct / total


@torch.no_grad()
def test_epoch(net, dataloader, device, time_steps=2):
    net.eval()
    correct = 0
    total = 0
    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)

        net.reset()
        out_fr = 0.0
        for t in range(time_steps):
            out_t = net(img, label=None)  # 测试时不提供 label
            out_fr += out_t

        pred = out_fr.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.numel()

    return 100.0 * correct / total


###############################
# 5) Optuna 的目标函数
###############################
def objective(trial: optuna.Trial):
    """
    搜索以下超参数:
      1) batch_size
      2) h1, h2 (两层隐藏层宽度)
      3) lr (label-driven STDP 学习率)
      4) time_steps
      5) tau_lif
    """
    # 搜索空间
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    h1 = trial.suggest_int("h1", 256, 1024, step=256)
    h2 = trial.suggest_int("h2", 128, 512, step=128)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    time_steps = trial.suggest_int("time_steps", 2, 8, step=2)
    tau_lif = trial.suggest_float("tau_lif", 1.0, 5.0)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, num_workers=2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构建更宽的多层网络
    net = WiderLabelSTDPNet(h1=h1, h2=h2, lr=lr, tau_lif=tau_lif).to(device)

    EPOCHS = 5  # 演示只跑5个epoch

    best_acc = 0.0
    for epoch in range(EPOCHS):
        train_acc = train_epoch(net, train_loader, device, time_steps)
        test_acc = test_epoch(net, test_loader, device, time_steps)
        best_acc = max(best_acc, test_acc)

        print(
            f"[Trial #{trial.number}] Epoch={epoch + 1}/{EPOCHS} "
            f"batch={batch_size}, h1={h1}, h2={h2}, lr={lr:.1e}, steps={time_steps}, tau={tau_lif:.2f} "
            f"=> train={train_acc:.2f}%, test={test_acc:.2f}%"
        )

        # prune
        trial.report(test_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_acc


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)  # 可改大 n_trials，但会很耗时

    print("\n===== Optuna 搜索结束 =====")
    print(f"Best trial idx: {study.best_trial.number}")
    print("Best hyperparams:", study.best_params)
    print(f"Best test_acc: {study.best_value:.2f}%")


if __name__ == "__main__":
    main()
