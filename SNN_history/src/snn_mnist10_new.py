import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# snnTorch
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
from snntorch import functional as SF

# ==============================
#         超参数
# ==============================
batch_size = 64
num_epochs = 2  # 演示用，可以多训练几轮
learning_rate = 1e-3
num_steps = 10  # SNN 的时间步
beta = 0.9  # LIF 膜电位衰减因子

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
#       数据加载
# ==============================
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 均值、方差
    ]
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
#  定义网络 (CNN + LIF)
# ==============================
# snnTorch 提供了神经元模块：snn.LIF() 可以直接调用 LIF 神经元
lif_params = {
    "beta": beta,  # 膜电位衰减因子
    "threshold": 1.0,  # 触发阈值
    "surrogate_function": surrogate.fast_sigmoid(),
    # 近似梯度，fast_sigmoid 等同于 1 / (1 + exp(-x)) 的一种近似
    "learn_threshold": False,  # 是否让 threshold 可训练
}


class SNN_CNN(nn.Module):
    def __init__(self):
        super(SNN_CNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # 对应的 LIF 神经元
        self.lif1 = snn.Leaky(**lif_params)  # 卷积后接
        self.lif2 = snn.Leaky(**lif_params)  # 全连接后接
        self.lif3 = snn.Leaky(**lif_params)  # 输出层

    def forward(self, x, num_steps=10):
        """
        在一个静态图像上重复 num_steps 次，模拟时间维度。
        逐步将激活值输入 LIF 神经元，累加脉冲输出。
        """
        # output_spikes 用于累加时间维度上的脉冲计数
        spk2_sum = 0
        for step in range(num_steps):
            # 同一个图像 x，多次输入
            cur_x = x

            # 卷积部分
            cur_x = self.conv1(cur_x)
            cur_x = self.lif1(cur_x)  # 第一层 LIF，输出就是脉冲或接近脉冲的激活
            cur_x = self.pool(cur_x)

            cur_x = self.conv2(cur_x)
            cur_x = self.lif2(cur_x)
            cur_x = self.pool(cur_x)

            # flatten
            cur_x = cur_x.view(cur_x.size(0), -1)
            cur_x = self.fc1(cur_x)
            cur_x = self.lif2(cur_x)  # 注意这里也可以新定义一个 LIF

            cur_x = self.fc2(cur_x)
            spk2 = self.lif3(cur_x)  # 输出层 LIF 的脉冲

            # 累加脉冲 (spk2是维度 [batch_size, 10])
            spk2_sum += spk2

        # 平均脉冲率作为分类分数
        out = spk2_sum / num_steps
        return out


# ==============================
#   模型初始化 & 优化器
# ==============================
model = SNN_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# ==============================
#   训练函数
# ==============================
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        # 前向传播
        out = model(data, num_steps)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += pred.eq(target).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(train_loader)
    acc = 100.0 * correct / len(train_loader.dataset)
    print(f"Train Epoch: {epoch}, Average Loss: {avg_loss:.6f}, Accuracy: {acc:.2f}%")


# ==============================
#   测试函数
# ==============================
def test(model, device, test_loader):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data, num_steps)
            loss = criterion(out, target)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(test_loader)
    acc = 100.0 * correct / len(test_loader.dataset)
    print(f"Test Set: Average Loss: {avg_loss:.6f}, Accuracy: {acc:.2f}%")


# ==============================
#        正式开始训练
# ==============================
if __name__ == "__main__":
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
