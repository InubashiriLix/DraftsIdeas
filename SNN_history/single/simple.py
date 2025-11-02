import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math


# ============ 1) 数据集：CIFAR-10(彩色) ============
def get_cifar10_dataloaders(batch_size=64):
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


# ============ 2) 替代梯度的自定义函数 ============
class SurrGradSpike(torch.autograd.Function):
    """
    前向: hard step (spike)
    反向: 用近似梯度
    """

    @staticmethod
    def forward(ctx, input_):
        # 前向就是Heaviside
        out = (input_ > 0).float()
        # 保存输入，以备 backward
        ctx.save_for_backward(input_)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [batch_size, ...], 是下游传回的梯度
        (input_,) = ctx.saved_tensors
        # 这里采用最简单的 triangular surrogate: d/dx = alpha * max(0, 1 - |x|)
        # 你也可以换成 sigmoid, step等方式
        alpha = 0.3  # 可调
        grad_input = grad_output.clone()
        mask = (input_.abs() < 1.0).float()  # |x|<1时则有梯度
        # 这里乘以 alpha*(1 - |x|)
        grad = alpha * mask * (1 - input_.abs())

        return grad_input * grad, None


# ============ 3) LIF 层(带替代梯度) ============
class SurrogateLIFLayer(nn.Module):
    """
    一个简单的 LIF 神经元层，带膜电位和发放逻辑。
    发放用上面的 SurrGradSpike 做反向传播近似。
    """

    def __init__(self, size, v_threshold=1.0, v_reset=0.0, decay=0.3):
        super().__init__()
        self.size = size
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.decay = decay

    def forward(self, input_, v):
        """
        input_: [batch_size, size]
        v: 上一时刻的膜电位 [batch_size, size]
        返回: spike, new_v
        """
        # 先对 v 做衰减
        v = v * self.decay + input_
        # 计算是否发放
        # 这里用 SurrGradSpike 来保证反向可导
        spike = SurrGradSpike.apply(v - self.v_threshold)
        # 对发放的神经元做 reset
        v = torch.where(spike > 0, torch.ones_like(v) * self.v_reset, v)
        return spike, v


# ============ 4) 网络：Flatten -> Linear -> SurrogateLIF -> Linear(输出) ============
class SurrogateLIFNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.lif = SurrogateLIFLayer(
            size=hidden_size, v_threshold=1.0, v_reset=0.0, decay=0.3
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, T=10):
        """
        多步仿真:
          x: [batch_size, 3, 32, 32]
          T: 时间步数
        返回：logits: [batch_size, output_size]
        """
        batch_size = x.shape[0]
        # flatten
        x = x.view(batch_size, -1)  # [batch_size, 3072]

        # 初始化膜电位
        v = torch.zeros(batch_size, self.lif.size, device=x.device)

        # 我们可以把最后一层当连续输出，也可以累计 hidden spikes
        # 这里示例：把最后一层的激活在每个时间步都算一次，然后取最后步的输出
        # 也可改成对输出求平均或累加
        out = None

        for t in range(T):
            # 先过fc_in
            h = self.fc_in(x)
            # lif层
            s, v = self.lif(h, v)  # spike
            # 输出层
            out_t = self.fc_out(s)
            # 我们这儿取"最后时刻"的 out 来分类（也可以做累加/平均）
            out = out_t

        return out


# ============ 5) 训练 & 测试循环 ============
def train_one_epoch(model, train_loader, optimizer, device, T):
    model.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # 前向
        out = model(images, T=T)  # [batch_size, 10]
        loss = F.cross_entropy(out, labels)

        # 反向
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item() * images.size(0)
        pred = out.argmax(dim=1)
        correct = (pred == labels).sum().item()
        total_correct += correct
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


def test_model(model, test_loader, device, T):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images, T=T)
            loss = F.cross_entropy(out, labels)

            total_loss += loss.item() * images.size(0)
            pred = out.argmax(dim=1)
            correct = (pred == labels).sum().item()
            total_correct += correct
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


# ============ 6) 主函数 ============
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=64)

    # 超参
    input_size = 3 * 32 * 32  # CIFAR-10 彩色
    hidden_size = 512
    output_size = 10
    T = 10  # 时间步
    lr = 1e-3
    epochs = 10

    # 构建模型
    model = SurrogateLIFNetwork(input_size, hidden_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, T
        )
        test_loss, test_acc = test_model(model, test_loader, device, T)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
            f"| Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )
