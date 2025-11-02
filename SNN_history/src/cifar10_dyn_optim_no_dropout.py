import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# ========== 0. 基础配置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 超参数
TIME_STEPS = 16  # 你可以尝试改大，如16、32
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
THRESH = 0.9  # LIF 阈值
DECAY = 0.3  # LIF 膜电位衰减

# ========== 1. 数据加载 (CIFAR-10 灰度 + Normalize) ==========
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
def binary_encode(tensor, time_steps=TIME_STEPS):
    temp = (tensor + 1.0) / 2.0 * 255.0
    temp = temp.clamp(0, 255).byte()

    batch_size, channels, H, W = temp.shape
    encoded = torch.zeros(batch_size, time_steps, channels, H, W, device=temp.device)

    for t in range(min(time_steps, 8)):
        bit_shift = 7 - t
        bit = (temp >> bit_shift) & 1
        encoded[:, t] = bit.float()

    return encoded


# ========== 3. 替代梯度 (线性窗) ==========
class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = (x > 0).float()
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = 1.0
        mask = (x.abs() < 1.0).float()
        grad_input = grad_output * alpha * mask
        return grad_input


# ========== 4. LIF 神经元 ==========
class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.2):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.spike_fn = SurrogateHeaviside.apply

    def forward(self, x, mem):
        mem = mem + x - self.decay * mem
        spike = self.spike_fn(mem - self.threshold)
        mem = mem * (1.0 - spike)
        return spike, mem


# ========== 5. 4 卷积层的 SNN 网络（去除 Dropout） ==========
class SurrogateBinarySNN_4Conv(nn.Module):
    def __init__(self, time_steps=8):
        super().__init__()
        self.time_steps = time_steps

        # 第1卷积: 1 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        # self.drop1 = nn.Dropout2d(p=drop_prob)  # 已去掉

        # 第2卷积: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # self.drop2 = nn.Dropout2d(p=drop_prob)

        # 第3卷: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        # self.drop3 = nn.Dropout2d(p=drop_prob)

        # 第4卷: 128 -> 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        # self.drop4 = nn.Dropout2d(p=drop_prob)

        # 全连接
        self.fc = nn.Linear(256 * 2 * 2, 10)

        # LIF 神经元
        self.lif1 = LIFNeuron(threshold=THRESH, decay=DECAY)
        self.lif2 = LIFNeuron(threshold=THRESH, decay=DECAY)
        self.lif3 = LIFNeuron(threshold=THRESH, decay=DECAY)
        self.lif4 = LIFNeuron(threshold=THRESH, decay=DECAY)
        self.lif_fc = LIFNeuron(threshold=THRESH, decay=DECAY)

    def forward(self, x):
        batch_size = x.size(0)
        mem1 = torch.zeros(batch_size, 32, 32, 32, device=x.device)
        mem2 = torch.zeros(batch_size, 64, 16, 16, device=x.device)
        mem3 = torch.zeros(batch_size, 128, 8, 8, device=x.device)
        mem4 = torch.zeros(batch_size, 256, 4, 4, device=x.device)
        mem_fc = torch.zeros(batch_size, 10, device=x.device)

        output_sum = torch.zeros(batch_size, 10, device=x.device)

        for t in range(self.time_steps):
            current_input = x[:, t]

            # 第1层: conv -> lif -> pool
            c1_out = self.conv1(current_input)
            s1, mem1 = self.lif1(c1_out, mem1)
            p1_out = self.pool1(s1)

            # 第2层
            c2_out = self.conv2(p1_out)
            s2, mem2 = self.lif2(c2_out, mem2)
            p2_out = self.pool2(s2)

            # 第3层
            c3_out = self.conv3(p2_out)
            s3, mem3 = self.lif3(c3_out, mem3)
            p3_out = self.pool3(s3)

            # 第4层
            c4_out = self.conv4(p3_out)
            s4, mem4 = self.lif4(c4_out, mem4)
            p4_out = self.pool4(s4)

            # 全连接
            flatten = p4_out.view(batch_size, -1)
            fc_out = self.fc(flatten)
            s_fc, mem_fc = self.lif_fc(fc_out, mem_fc)

            output_sum += fc_out

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

    return acc


# ========== 7. 主函数 (增加学习率调度) ==========
if __name__ == "__main__":
    model = SurrogateBinarySNN_4Conv(time_steps=TIME_STEPS).to(device)
    criterion = nn.CrossEntropyLoss()

    # 优化器 & 学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 120, 160], gamma=0.2
    )
    #   milestones: 在这几个 epoch 调整学习率
    #   gamma: 每次到达里程碑时，学习率 * gamma

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, criterion, epoch)
        test_acc = test(model, test_loader, criterion, epoch)

        # 每个 epoch 结束后调用调度器, 让学习率随 epoch 调度
        scheduler.step()

        # 如果本 epoch 测试集准确率优于之前最好纪录，则保存
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model_dyn_optim.pth")
            print(
                f"==> New best accuracy: {best_acc:.2f}% at epoch {best_epoch}, model saved!"
            )

    print(
        f"Training completed. Best test accuracy: {best_acc:.2f}% (epoch {best_epoch})."
    )
