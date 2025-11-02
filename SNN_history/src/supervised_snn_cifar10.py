import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

# ========== 0. 配置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 超参数
TIME_STEPS = 8
BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-3  # 降低初始学习率
WD = 1e-4  # Weight Decay
DROP_PROB = 0.1  # Dropout 概率
BETA = 5.0  # Surrogate平滑系数

# ========== 1. 数据 (CIFAR-10 灰度 + 数据增强 + Normalize) ==========
transform_train = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)


# ========== 2. 二进制脉冲编码函数 ==========
def binary_encode(tensor, time_steps=8):
    """
    将灰度图像[-1,1]转为8位二进制脉冲.
    输入: [batch,1,H,W], 输出: [batch,time_steps,1,H,W]
    """
    temp = (tensor + 1.0) / 2.0 * 255.0
    temp = temp.clamp(0, 255).byte()

    batch_size, channels, H, W = temp.shape
    encoded = torch.zeros(batch_size, time_steps, channels, H, W, device=temp.device)

    for t in range(time_steps):
        bit_shift = 7 - t
        bit = (temp >> bit_shift) & 1
        encoded[:, t] = bit.float()

    return encoded


# ========== 3. 替代梯度 (Sigmoid 近似) ==========
class SurrogateSigmoidHeaviside(torch.autograd.Function):
    """
    前向: (x>0)->1 否则0
    反向: sigmoid 的梯度近似
    """

    @staticmethod
    def forward(ctx, x):
        out = (x > 0).float()
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        beta = BETA
        s = 1 / (1 + torch.exp(-beta * x))
        grad_input = grad_output * beta * s * (1.0 - s)
        return grad_input


# ========== 4. LIF 神经元 ==========
class LIFNeuron(nn.Module):
    """
    膜电位 + 阈值 + 重置, 使用 Sigmoid surrogate
    """

    def __init__(self, threshold=1.0, decay=0.2):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.spike_fn = SurrogateSigmoidHeaviside.apply

    def forward(self, x, mem):
        mem = mem + x - self.decay * mem
        spike = self.spike_fn(mem - self.threshold)
        mem = mem * (1.0 - spike)
        return spike, mem


# ========== 5. 深度 SNN(去掉 BN, 加 Dropout) ==========
class DeepSNN_Dropout(nn.Module):
    def __init__(self, time_steps=8, drop_prob=0.3):
        super().__init__()
        self.time_steps = time_steps

        # 第1层: conv -> LIF -> pool -> dropout
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.lif1 = LIFNeuron(threshold=0.9, decay=0.2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=drop_prob)

        # 第2层: conv -> LIF -> pool -> dropout
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = LIFNeuron(threshold=0.9, decay=0.2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(p=drop_prob)

        # 第3层: conv -> LIF -> pool -> dropout
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif3 = LIFNeuron(threshold=0.9, decay=0.2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(p=drop_prob)

        # 全连接
        self.fc = nn.Linear(128 * 4 * 4, 10)
        self.lif_fc = LIFNeuron(threshold=0.0, decay=0.2)

    def forward(self, x):
        """
        x: [batch, time_steps, 1, 32, 32]
        return: [batch, 10]
        """
        batch_size = x.size(0)
        # 初始化膜电位
        mem1 = torch.zeros(batch_size, 32, 32, 32, device=x.device)
        mem2 = torch.zeros(batch_size, 64, 16, 16, device=x.device)
        mem3 = torch.zeros(batch_size, 128, 8, 8, device=x.device)
        mem_fc = torch.zeros(batch_size, 10, device=x.device)

        output_sum = torch.zeros(batch_size, 10, device=x.device)

        for t in range(self.time_steps):
            current_input = x[:, t]  # [batch,1,32,32]

            # 第1层
            c1_out = self.conv1(current_input)
            s1, mem1 = self.lif1(c1_out, mem1)
            p1_out = self.pool1(s1)
            d1_out = self.drop1(p1_out)

            # 第2层
            c2_out = self.conv2(d1_out)
            s2, mem2 = self.lif2(c2_out, mem2)
            p2_out = self.pool2(s2)
            d2_out = self.drop2(p2_out)

            # 第3层
            c3_out = self.conv3(d2_out)
            s3, mem3 = self.lif3(c3_out, mem3)
            p3_out = self.pool3(s3)
            d3_out = self.drop3(p3_out)

            # 全连接
            flatten = d3_out.view(batch_size, -1)
            fc_out = self.fc(flatten)
            s_fc, mem_fc = self.lif_fc(fc_out, mem_fc)

            output_sum += fc_out

        # 输出
        return output_sum / self.time_steps


# ========== 6. 训练 & 测试函数 ==========
def train(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        encoded = binary_encode(data, time_steps=TIME_STEPS)

        optimizer.zero_grad()
        outputs = model(encoded)
        loss = criterion(outputs, target)
        loss.backward()
        # 可选梯度裁剪: nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

    acc = 100.0 * correct / total
    print(
        f"Epoch {epoch} - Train Loss: {running_loss / len(loader):.4f}, Train Acc: {acc:.2f}%"
    )


def test(model, loader, criterion, epoch):
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            encoded = binary_encode(data, time_steps=TIME_STEPS)
            outputs = model(encoded)
            loss = criterion(outputs, target)
            loss_total += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

    acc = 100.0 * correct / total
    print(
        f"Epoch {epoch} - Test Loss: {loss_total / len(loader):.4f}, Test Acc: {acc:.2f}%"
    )


# ========== 7. 主函数 ==========
if __name__ == "__main__":
    model = DeepSNN_Dropout(time_steps=TIME_STEPS, drop_prob=DROP_PROB).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # 学习率调度: 每隔30个epoch将lr乘以0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, criterion, epoch)
        test(model, test_loader, criterion, epoch)
        scheduler.step()  # 更新学习率
