import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ==============================
#        超参数定义
# ==============================
batch_size = 64
learning_rate = 1e-3
num_epochs = 300  # 可酌情减少
time_window = 10
threshold = 1.0
decay = 0.9
soft_reset = True

drop_out_value = 0.2

# Surrogate Gradient 超参数
surrogate_type = "piecewise"  # 可改 "fast_sigmoid" 等
alpha = 0.3
grad_scale = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
#        数据准备 (CIFAR-10)
#        不转灰度、增加数据增强
# ==============================
train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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


# (以下略去 STBPSurrogateFunction, stbp_spike_function, ImprovedLIFNeuron 保持不变)
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


def stbp_spike_function(x):
    return STBPSurrogateFunction.apply(x)


# ==============================
#   网络结构定义 (CNN + LIF)
#   - 改回3输入
#   - 仅加一个Dropout(0.3)
# ==============================
class SNN_CNN(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9, soft_reset=True):
        super(SNN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3通道输入
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(drop_out_value)  # 简单Dropout
        self.fc2 = nn.Linear(256, 10)

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
        self.lif1.mem = None
        self.lif2.mem = None
        self.lif3.mem = None
        self.lif4.mem = None

        out_spike_sum = 0
        for _ in range(time_window):
            out = self.conv1(x)
            out = nn.ReLU()(out)
            spike1 = self.lif1(out)

            out = self.conv2(spike1)
            out = nn.ReLU()(out)
            spike2 = self.lif2(out)
            out = self.pool(spike2)

            out = self.conv3(out)
            out = nn.ReLU()(out)
            spike3 = self.lif3(out)

            out = self.conv4(spike3)
            out = nn.ReLU()(out)
            spike4 = self.lif4(out)
            out = self.pool(spike4)

            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.dropout(out)  # Dropout抑制过拟合
            out = nn.ReLU()(out)
            out = self.fc2(out)
            out_spike_sum += out

        return out_spike_sum / time_window


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, time_window=time_window)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 省略中间打印，按需自行添加


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
    print(f"Test: Loss={avg_loss:.6f}, Accuracy={accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    model = SNN_CNN(threshold=threshold, decay=decay, soft_reset=soft_reset).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 不加weight_decay

    # 学习率衰减(可选)
    # from torch.optim.lr_scheduler import MultiStepLR
    # scheduler = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)

        # 提前停顿或减少epoch的逻辑
        # if acc > best_acc:
        #     best_acc = acc
        #     torch.save(model.state_dict(), "best.pth")
        # if scheduler is not None:
        #     scheduler.step()
