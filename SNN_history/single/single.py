import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import optuna
import numpy as np

# ------------------------------------------------------
# 一些默认超参 (当不使用optuna时可用, 或仅作示例)
# ------------------------------------------------------
DEFAULT_CONFIG = {
    "hidden_size": 256,
    "batch_size": 32,
    "T": 10,  # 仿真时间步数
    "lr_stdp": 1e-3,  # STDP学习率
    "epochs": 10,  # 训练 epoch 数
    "lr_classifier": 1e-3,  # 读出层的优化器学习率
}


# ------------------------------------------------------
# 数据集准备: CIFAR-10 保留彩色, 常规增广
# ------------------------------------------------------
def get_cifar10_dataloaders(batch_size=64):
    """
    返回 train_loader 和 test_loader，用于CIFAR-10彩色图像。
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ------------------------------------------------------
# 简易 LIF 神经元层
# ------------------------------------------------------
class LIFNeuronLayer:
    """
    用于在前向计算时保存膜电位和发放脉冲(spike)。
    """

    def __init__(self, size, v_threshold=1.0, v_reset=0.0, decay=0.3):
        """
        size: 该层神经元数量
        v_threshold: 膜电位阈值
        v_reset: 膜电位重置值
        decay: 简易衰减因子
        """
        self.size = size
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.decay = decay

        self.v = None  # 膜电位

    def init_state(self, batch_size, device):
        """
        初始化膜电位
        """
        self.v = torch.zeros(batch_size, self.size, device=device)

    def forward(self, input_current):
        """
        input_current: [batch_size, size]
        返回：spike (0/1)
        """
        # 衰减
        self.v = self.v * self.decay
        # 累加输入
        self.v += input_current
        # 判断发放脉冲
        spikes = (self.v >= self.v_threshold).float()
        # 重置
        self.v = torch.where(spikes > 0, torch.ones_like(self.v) * self.v_reset, self.v)
        return spikes


# ------------------------------------------------------
# 简单全连接SNN: Flatten -> fc_in -> LIF -> fc_out
# ------------------------------------------------------
class SpikingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpikingNetwork, self).__init__()

        self.fc_in = nn.Linear(input_size, hidden_size, bias=False)
        self.lif = LIFNeuronLayer(size=hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size, bias=True)

        # 初始化权重
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, x, T=10):
        """
        x: [batch_size, 3, 32, 32]
        T: 仿真时间步
        返回：输出层的脉冲计数 [batch_size, output_size]
        """
        batch_size = x.shape[0]
        # Flatten (3*32*32 = 3072)
        x = x.view(batch_size, -1)

        self.lif.init_state(batch_size, x.device)
        out_spike_sum = torch.zeros(
            batch_size, self.fc_out.out_features, device=x.device
        )

        for _ in range(T):
            # 简单做rate编码 (随机)
            rand_vals = torch.rand_like(x)
            pre_spikes_t = (rand_vals < torch.sigmoid(x)).float()

            h = self.fc_in(pre_spikes_t)
            spikes = self.lif.forward(h)

            out = self.fc_out(spikes)
            out_spikes = (out > 0).float()
            out_spike_sum += out_spikes

        return out_spike_sum

    def get_in_weights(self):
        # 用于STDP更新
        return self.fc_in.weight


# ------------------------------------------------------
# 双向 STDP 更新函数
# ------------------------------------------------------
def stdp_update_dual(
    pre_spikes_curr,  # [batch_size, input_dim]  at time t
    post_spikes_curr,  # [batch_size, hidden_size] at time t
    pre_spikes_next,  # [batch_size, input_dim]  at time t+1
    post_spikes_next,  # [batch_size, hidden_size] at time t+1
    weights,  # [hidden_size, input_dim]
    lr_stdp,
    reward=1.0,
):
    """
    实现一个双向STDP：
      ΔW ∝ (post(t+1)⊗pre(t)) - (post(t)⊗pre(t+1))，再乘以reward。
    这里 pre⊗post 表示外积 => [batch_size, hidden_size, input_dim]。
    """
    # pos_dw: 当 pre(t) & post(t+1) 同时发放，权重增强
    pos_dw = torch.bmm(
        post_spikes_next.unsqueeze(2),  # [batch_size, hidden_size, 1]
        pre_spikes_curr.unsqueeze(1),  # [batch_size, 1, input_dim]
    )  # => [batch_size, hidden_size, input_dim]

    # neg_dw: 当 post(t) & pre(t+1) 同时发放，权重减弱
    neg_dw = torch.bmm(
        post_spikes_curr.unsqueeze(2), pre_spikes_next.unsqueeze(1)
    )  # => [batch_size, hidden_size, input_dim]

    dw = (pos_dw - neg_dw).sum(dim=0)  # => [hidden_size, input_dim]
    dw = dw * lr_stdp * reward
    weights.data += dw


# ------------------------------------------------------
# 训练：在训练循环里手动获取各时刻 pre/post spike 并执行双向STDP
# ------------------------------------------------------
def train_one_epoch_stdp(model, train_loader, optimizer_out, device, config):
    model.train()
    T = config["T"]
    lr_stdp = config["lr_stdp"]

    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.shape[0]

        optimizer_out.zero_grad()

        # 手动时序展开
        x = images.view(batch_size, -1)  # [batch_size, 3*32*32 = 3072]
        model.lif.init_state(batch_size, device)

        pre_spikes_list = []
        post_spikes_list = []

        out_spike_sum = torch.zeros(
            batch_size, model.fc_out.out_features, device=device
        )

        for t in range(T):
            # 构建输入层脉冲 (rate编码)
            rand_vals = torch.rand_like(x)
            pre_spikes_t = (rand_vals < torch.sigmoid(x)).float()
            pre_spikes_list.append(pre_spikes_t)

            # 输入 -> hidden
            h_t = model.fc_in(pre_spikes_t)
            post_spikes_t = model.lif.forward(h_t)
            post_spikes_list.append(post_spikes_t)

            # hidden -> output
            out_t = model.fc_out(post_spikes_t)
            out_spikes_t = (out_t > 0).float()
            out_spike_sum += out_spikes_t

        # 计算预测
        _, pred = out_spike_sum.max(dim=1)
        correct = (pred == labels).sum().item()
        total_correct += correct
        total_samples += batch_size

        # reward: 整个batch的平均正确率决定正负
        batch_reward = torch.where(pred == labels, 1.0, -1.0).to(device)
        reward_val = batch_reward.mean().item()

        # 双向STDP
        W_in = model.get_in_weights()
        for t in range(T - 1):
            stdp_update_dual(
                pre_spikes_curr=pre_spikes_list[t],
                post_spikes_curr=post_spikes_list[t],
                pre_spikes_next=pre_spikes_list[t + 1],
                post_spikes_next=post_spikes_list[t + 1],
                weights=W_in,
                lr_stdp=lr_stdp,
                reward=reward_val,
            )

        # hidden->输出 做常规交叉熵反传
        out_spike_sum.requires_grad_()
        loss = F.cross_entropy(out_spike_sum, labels)
        loss.backward()
        optimizer_out.step()

    return 100.0 * total_correct / total_samples


# ------------------------------------------------------
# 测试
# ------------------------------------------------------
def test_model(model, test_loader, device, config):
    model.eval()
    T = config["T"]

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]

            out_spike_sum = torch.zeros(
                batch_size, model.fc_out.out_features, device=device
            )

            # 时序forward
            x = images.view(batch_size, -1)
            model.lif.init_state(batch_size, x.device)

            for _ in range(T):
                rand_vals = torch.rand_like(x)
                pre_spikes_t = (rand_vals < torch.sigmoid(x)).float()
                h_t = model.fc_in(pre_spikes_t)
                post_spikes_t = model.lif.forward(h_t)
                out_t = model.fc_out(post_spikes_t)
                out_spike_sum += (out_t > 0).float()

            _, pred = out_spike_sum.max(dim=1)
            correct = (pred == labels).sum().item()
            total_correct += correct
            total_samples += batch_size

    return 100.0 * total_correct / total_samples


# ------------------------------------------------------
# Optuna 超参数搜索
# ------------------------------------------------------
def objective(trial):
    """
    这里示例仍然给出一个超参搜索空间, 你可根据需求修改:
      - hidden_size: [128, 1024], step=128
      - batch_size:  [32, 128],  step=32
      - T:           [5, 25],    step=5
      - lr_stdp:     [1e-5, 1e-2], log-scale
      - epochs:      [5, 20]
      - lr_classifier: [1e-4, 1e-2], log-scale
    """
    hidden_size = trial.suggest_int("hidden_size", 128, 1024, step=128)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    T = trial.suggest_int("T", 5, 25, step=5)
    lr_stdp = trial.suggest_float("lr_stdp", 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 5, 20)
    lr_classifier = trial.suggest_float("lr_classifier", 1e-4, 1e-2, log=True)

    # 准备数据
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型 (彩色输入时 input_size=3*32*32=3072)
    model = SpikingNetwork(
        input_size=3 * 32 * 32, hidden_size=hidden_size, output_size=10
    ).to(device)

    # 优化器（针对 fc_out）
    optimizer_out = optim.Adam(model.fc_out.parameters(), lr=lr_classifier)

    # 配置
    config = {
        "T": T,
        "lr_stdp": lr_stdp,
        "epochs": epochs,
    }

    best_val_acc = 0.0
    for epoch in range(epochs):
        train_acc = train_one_epoch_stdp(
            model, train_loader, optimizer_out, device, config
        )
        val_acc = test_model(model, test_loader, device, config)

        # 记录最好结果
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Optuna的中途监控
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc


def run_optuna_search(n_trials=10):
    """
    运行Optuna搜索, n_trials可以自行调整
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n========== Optuna Search Completed ==========")
    print("Best Trial:")
    print(study.best_trial)
    print("Best Hyperparams:")
    print(study.best_params)
    print("Best Val Acc: {:.2f}%".format(study.best_value))


# ------------------------------------------------------
# 主函数
# ------------------------------------------------------
if __name__ == "__main__":
    print("shit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 使用默认配置先跑一个演示
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=DEFAULT_CONFIG["batch_size"]
    )

    model = SpikingNetwork(
        input_size=3 * 32 * 32,  # 彩色图像
        hidden_size=DEFAULT_CONFIG["hidden_size"],
        output_size=10,
    ).to(device)
    optimizer_out = optim.Adam(
        model.fc_out.parameters(), lr=DEFAULT_CONFIG["lr_classifier"]
    )

    print("开始使用默认配置训练 (彩色输入)...")
    for epoch in range(DEFAULT_CONFIG["epochs"]):
        train_acc = train_one_epoch_stdp(
            model, train_loader, optimizer_out, device, DEFAULT_CONFIG
        )
        test_acc = test_model(model, test_loader, device, DEFAULT_CONFIG)
        print(
            f"Epoch [{epoch + 1}/{DEFAULT_CONFIG['epochs']}], "
            f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%"
        )

    # 2) 使用 Optuna 搜索更优超参
    print("\n开始使用Optuna搜索超参数...")
    run_optuna_search(n_trials=5)
