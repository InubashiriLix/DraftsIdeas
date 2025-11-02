import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ==============================
#        超参数定义
# ==============================
batch_size = 64
learning_rate = 1e-3
num_epochs = 2  # 演示用 epoch 较少，实际可加大
time_window = 10  # 时间步数
threshold = 1.0  # LIF 神经元触发阈值
decay = 0.9  # LIF 神经元膜电位衰减系数
soft_reset = True  # True=软重置, False=硬重置

# Surrogate Gradient超参数 (用于 STBP 思路)
# 这里只是演示，你可以不断试各种近似函数
surrogate_type = "piecewise"  # 可选 "piecewise" 或 "fast_sigmoid"
alpha = 0.3  # 分段线性时的斜率
grad_scale = 1.0  # 缩放项（可根据经验或再调）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
#        数据准备
# ==============================
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)


# ==============================
#   改进版 Surrogate Gradient
# ==============================
class STBPSurrogateFunction(torch.autograd.Function):
    """
    支持多种近似梯度的自定义脉冲函数，用于 STBP。
    forward:  output = 1.0 if x>0 else 0
    backward: d(output)/d(x) ~ surrogate(x)
    """

    @staticmethod
    def forward(ctx, input):
        # 触发脉冲：大于0视为脉冲(1)，否则(0)
        out = (input > 0).float()
        # 保存用于 backward
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = None

        # 这里可自行定义各种近似函数
        if surrogate_type == "piecewise":
            # 分段线性：在 |x| <= 1 内给定斜率 alpha，在之外梯度=0
            mask = (input.abs() <= 1.0).float()
            grad_input = grad_output * mask * alpha * grad_scale

        elif surrogate_type == "fast_sigmoid":
            # fast sigmoid 近似: d/dx = alpha * (1 / (1 + |x|))^2
            # 也有人定义成 alpha*(1 - |x|/2)^2 之类，都差不多
            # 这里随便举一个例子
            tmp = 1.0 / (1.0 + input.abs())
            grad = (alpha * (tmp**2)) * grad_scale
            grad_input = grad_output * grad

        else:
            # 默认最简单的直通梯度: d/dx=1 in [-1,1], else=0
            mask = (input.abs() <= 1.0).float()
            grad_input = grad_output * mask * grad_scale

        return grad_input


# 为了使用起来更简洁，可以直接写个函数
def stbp_spike_function(x):
    return STBPSurrogateFunction.apply(x)


# ==============================
#   改进版 LIF 神经元定义
# ==============================
class ImprovedLIFNeuron(nn.Module):
    """
    改进版本的 LIF，包含:
      - 可选硬重置/软重置
      - 使用更丰富的 Surrogate Gradient (STBP)
      - 其余参数见注释
    """

    def __init__(self, threshold=1.0, decay=0.9, soft_reset=True):
        super(ImprovedLIFNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay
        self.soft_reset = soft_reset
        self.mem = None  # 膜电位

    def forward(self, x):
        # 如果是第一次 forward 或 batch_size 改变，就把 mem 置为 0
        if self.mem is None or self.mem.shape[0] != x.shape[0]:
            self.mem = torch.zeros_like(x).to(x.device)

        # 更新膜电位
        self.mem = self.mem * self.decay + x

        # 判断是否触发脉冲
        spike = stbp_spike_function(self.mem - self.threshold)
        if spike is None:
            raise Exception("Spike function returned None!")

        # 重置膜电位
        if self.soft_reset:
            # 软重置: mem = mem - Vth (只在触发位置执行)
            self.mem = self.mem - spike * self.threshold
        else:
            # 硬重置: mem = 0 (只在触发位置执行)
            self.mem = self.mem * (1 - spike)

        return spike


# ==============================
#   网络结构定义 (CNN + 改进LIF)
# ==============================
class SNN_CNN(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9, soft_reset=True):
        super(SNN_CNN, self).__init__()
        # 卷积提取特征
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # 改进版 LIF 神经元
        self.lif1 = ImprovedLIFNeuron(
            threshold=threshold, decay=decay, soft_reset=soft_reset
        )
        self.lif2 = ImprovedLIFNeuron(
            threshold=threshold, decay=decay, soft_reset=soft_reset
        )

    def forward(self, x, time_window=10):
        # 每次 forward 前重置膜电位
        self.lif1.mem = None
        self.lif2.mem = None

        out_spike_sum = 0

        for t in range(time_window):
            # 静态图像，多次输入
            out = self.conv1(x)
            out = nn.ReLU()(out)
            out = self.pool(out)

            out = self.conv2(out)
            out = nn.ReLU()(out)
            out = self.pool(out)

            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = nn.ReLU()(out)

            # 第一个 LIF
            spike1 = self.lif1(out)

            # 全连接 -> 第二个 LIF
            out = self.fc2(spike1)
            spike2 = self.lif2(out)

            # 累加脉冲
            out_spike_sum += spike2

        # 返回平均脉冲率作为分类输出
        return out_spike_sum / time_window


# ==============================
#       训练与测试流程
# ==============================
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        # 前向传播
        output = model(data, time_window=time_window)
        loss = criterion(output, target)
        loss.backward()  # 反向传播 (STBP 通过我们的 SurrogateFunction 在此发挥作用)
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / len(train_loader.dataset)
    print(
        f"Train Epoch: {epoch}  Average loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%"
    )


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, time_window=time_window)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%")


# ==============================
#        运行入口
# ==============================
if __name__ == "__main__":
    model = SNN_CNN(threshold=threshold, decay=decay, soft_reset=soft_reset).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
