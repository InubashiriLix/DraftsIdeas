import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ==============================
#        超参数定义
# ==============================
batch_size = 64
learning_rate = 1e-3
num_epochs = 200  # 演示用，可视情况增大
time_window = 10  # 时间步数 (SNN 展开)
threshold = 1.0  # LIF 神经元触发阈值
decay = 0.9  # 膜电位衰减系数
soft_reset = True  # True=软重置, False=硬重置

# Surrogate Gradient 超参数
surrogate_type = "piecewise"  # 可改 "fast_sigmoid" 等
alpha = 0.3
grad_scale = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
#        数据准备 (CIFAR-10) -> 转灰度
# ==============================
# 将彩色的 CIFAR-10 转为灰度单通道图像
# 注意这里的随机裁剪、水平翻转等数据增强仍然保留；如果不需要，可以删除
train_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # 灰度下的均值/方差可自行设定，这里简单设成 0.5 / 0.5
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
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
    @staticmethod
    def forward(ctx, input):
        out = (input > 0).float()
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = None

        if surrogate_type == "piecewise":
            # 在 |x| <= 1 内梯度= alpha，否则=0
            mask = (input.abs() <= 1.0).float()
            grad_input = grad_output * mask * alpha * grad_scale

        elif surrogate_type == "fast_sigmoid":
            # 举例： alpha * (1/(1 + |x|))^2
            tmp = 1.0 / (1.0 + input.abs())
            grad = alpha * (tmp**2) * grad_scale
            grad_input = grad_output * grad

        else:
            # 默认：简单的直通梯度
            mask = (input.abs() <= 1.0).float()
            grad_input = grad_output * mask * grad_scale

        return grad_input


def stbp_spike_function(x):
    return STBPSurrogateFunction.apply(x)


# ==============================
#   改进版 LIF 神经元定义
# ==============================
class ImprovedLIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9, soft_reset=True):
        super(ImprovedLIFNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay
        self.soft_reset = soft_reset
        self.mem = None

    def forward(self, x):
        if self.mem is None or self.mem.shape[0] != x.shape[0]:
            self.mem = torch.zeros_like(x).to(x.device)

        # 更新膜电位
        self.mem = self.mem * self.decay + x

        # 判断是否触发脉冲
        spike = stbp_spike_function(self.mem - self.threshold)

        # 重置膜电位
        if self.soft_reset:
            self.mem = self.mem - spike * self.threshold
        else:
            self.mem = self.mem * (1 - spike)

        return spike


# ==============================
#   网络结构定义 (CNN + 改进LIF)
# ==============================
class SNN_CNN(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9, soft_reset=True):
        super(SNN_CNN, self).__init__()
        # 原本 CIFAR-10 是 3 通道(RGB)
        # 现在已经转为灰度 -> 1 通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 再来两层卷积
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # 全连接层 (after 2 pooling layers, feature map size = 8x8)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

        # 改进版 LIF 神经元
        self.lif1 = ImprovedLIFNeuron(
            threshold=threshold, decay=decay, soft_reset=soft_reset
        )
        self.lif2 = ImprovedLIFNeuron(
            threshold=threshold, decay=decay, soft_reset=soft_reset
        )
        self.lif3 = ImprovedLIFNeuron(
            threshold=threshold, decay=decay, soft_reset=soft_reset
        )
        self.lif4 = ImprovedLIFNeuron(
            threshold=threshold, decay=decay, soft_reset=soft_reset
        )

    def forward(self, x, time_window=10):
        # 每次 forward 前重置膜电位
        self.lif1.mem = None
        self.lif2.mem = None
        self.lif3.mem = None
        self.lif4.mem = None

        out_spike_sum = 0

        for _ in range(time_window):
            # 前两个卷积
            out = self.conv1(x)
            out = nn.ReLU()(out)
            spike1 = self.lif1(out)

            out = self.conv2(spike1)
            out = nn.ReLU()(out)
            spike2 = self.lif2(out)
            out = self.pool(spike2)

            # 后两个卷积
            out = self.conv3(out)
            out = nn.ReLU()(out)
            spike3 = self.lif3(out)

            out = self.conv4(spike3)
            out = nn.ReLU()(out)
            spike4 = self.lif4(out)
            out = self.pool(spike4)

            # flatten
            out = out.view(out.size(0), -1)

            # 全连接
            out = self.fc1(out)
            out = nn.ReLU()(out)
            out = self.fc2(out)

            # 累加脉冲(或激活)
            out_spike_sum += out

        # 平均结果作为分类 logits
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
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / len(train_loader.dataset)
    print(
        f"Train Epoch: {epoch}, Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%"
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
    print(f"Test Set: Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%")


# ==============================
#        运行入口
# ==============================
if __name__ == "__main__":
    model = SNN_CNN(threshold=threshold, decay=decay, soft_reset=soft_reset).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
