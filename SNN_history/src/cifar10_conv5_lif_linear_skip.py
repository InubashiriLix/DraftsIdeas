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
TIME_STEPS = 16
BATCH_SIZE = 64
EPOCHS = 1000
LR = 1e-3

THRESH = 0.9  # LIF 发放阈值
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
    """
    将输入图像转换为多比特(二进制)脉冲序列:
      - 把像素映射到[0,255],
      - 逐位拆成 bit,
      - 在前8个time_step内分别发放。
    """
    temp = (tensor + 1.0) / 2.0 * 255.0
    temp = temp.clamp(0, 255).byte()

    batch_size, channels, H, W = temp.shape
    encoded = torch.zeros(batch_size, time_steps, channels, H, W, device=temp.device)

    # 只对前8个time_step做二进制展开
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
    """
    mem(t+1) = mem(t) + x - decay*mem(t)
    spike(t+1) = Heaviside(mem(t+1) - threshold)
    若spike==1, 则 mem(t+1)=0
    """

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


# ========== 5. 5层卷积 + 跳连的SNN (conv->LIF->pool 结构) ==========
class SurrogateBinarySNN_5ConvRes(nn.Module):
    """
    结构：
      Layer1: conv1 -> LIF1 -> pool1
      Layer2: conv2 -> LIF2 -> pool2
      Layer3: conv3 -> LIF3 -> pool3
        - 输出 shape: [128, 4,4]
      Layer4: conv4 -> LIF4 (无pool,保持 4x4)
        - 同时用 skip_conv4 将(128,4,4)映射到(256,4,4)，加到 LIF4 的输出上
      Layer5: conv5 -> LIF5 -> pool5 (变为 [256,2,2])
      最后 fc -> LIF_fc -> 得到 [batch, 10]
    """

    def __init__(self, time_steps=8):
        super().__init__()
        self.time_steps = time_steps

        # 第1层: 1 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.lif1 = LIFNeuron(threshold=THRESH, decay=DECAY)

        # 第2层: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.lif2 = LIFNeuron(threshold=THRESH, decay=DECAY)

        # 第3层: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.lif3 = LIFNeuron(threshold=THRESH, decay=DECAY)

        # 第4层: 128 -> 256 (无pool)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.lif4 = LIFNeuron(threshold=THRESH, decay=DECAY)
        # skip: 用 1x1 卷积把 (128,4,4) -> (256,4,4)
        self.skip_conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)

        # 第5层: 256 -> 256 + pool
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.lif5 = LIFNeuron(threshold=THRESH, decay=DECAY)
        self.pool5 = nn.MaxPool2d(2, 2)  # 从 (4,4) -> (2,2)

        # 全连接 (256*2*2 -> 10)
        self.fc = nn.Linear(256 * 2 * 2, 10)
        self.lif_fc = LIFNeuron(threshold=THRESH, decay=DECAY)

    def forward(self, x):
        """
        x: [batch_size, time_steps, 1, 32, 32]
        """
        batch_size = x.size(0)

        # 对应各层 LIF 的膜电位初值 (跟 "conv -> LIF" 的输出形状一致)
        # layer1: conv1输出[32,32,32], lif1的mem1形状=[32,32,32]
        mem1 = torch.zeros(batch_size, 32, 32, 32, device=x.device)

        # layer2: conv2输出[64,16,16] (因为pool1后size=16)
        mem2 = torch.zeros(batch_size, 64, 16, 16, device=x.device)

        # layer3: conv3输出[128,8,8] (pool2后=8,8)
        mem3 = torch.zeros(batch_size, 128, 8, 8, device=x.device)

        # layer4: conv4输出[256,4,4] (pool3后=4,4)
        mem4 = torch.zeros(batch_size, 256, 4, 4, device=x.device)

        # layer5: conv5输出[256,4,4], pool5后= [256,2,2]
        mem5 = torch.zeros(batch_size, 256, 4, 4, device=x.device)

        # fc: [batch, 10]
        mem_fc = torch.zeros(batch_size, 10, device=x.device)

        output_sum = torch.zeros(batch_size, 10, device=x.device)

        for t in range(self.time_steps):
            # 取第 t 个时间步的输入脉冲
            current_input = x[:, t]  # [batch, 1, 32, 32]

            # ===== 第1层 (conv1 -> lif1 -> pool1) =====
            c1_out = self.conv1(current_input)  # [batch,32,32,32]
            s1, mem1 = self.lif1(c1_out, mem1)  # [batch,32,32,32]
            p1_out = self.pool1(s1)  # [batch,32,16,16]

            # ===== 第2层 (conv2 -> lif2 -> pool2) =====
            c2_out = self.conv2(p1_out)  # [batch,64,16,16]
            s2, mem2 = self.lif2(c2_out, mem2)  # [batch,64,16,16]
            p2_out = self.pool2(s2)  # [batch,64,8,8]

            # ===== 第3层 (conv3 -> lif3 -> pool3) =====
            c3_out = self.conv3(p2_out)  # [batch,128,8,8]
            s3, mem3 = self.lif3(c3_out, mem3)  # [batch,128,8,8]
            p3_out = self.pool3(s3)  # [batch,128,4,4]

            # ===== 第4层 (conv4 -> lif4) + skip =====
            # conv4 输入: [128,4,4], 输出 [256,4,4]
            c4_out = self.conv4(p3_out)  # [batch,256,4,4]
            s4, mem4 = self.lif4(c4_out, mem4)  # [batch,256,4,4]
            # skip 分支: [128,4,4] -> [256,4,4]
            skip_out = self.skip_conv4(p3_out)  # [batch,256,4,4]
            # 残差相加
            res4_out = s4 + skip_out  # [batch,256,4,4]

            # ===== 第5层 (conv5 -> lif5 -> pool5) =====
            c5_out = self.conv5(res4_out)  # [batch,256,4,4]
            s5, mem5 = self.lif5(c5_out, mem5)  # [batch,256,4,4]
            p5_out = self.pool5(s5)  # [batch,256,2,2]

            # ===== 全连接 =====
            flatten = p5_out.view(batch_size, -1)  # [batch,256*2*2]
            fc_out = self.fc(flatten)  # [batch,10]
            s_fc, mem_fc = self.lif_fc(fc_out, mem_fc)  # [batch,10]

            output_sum += fc_out  # 或者累加 s_fc 看情况

        # 时间步平均作为输出
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


# ========== 7. 主函数 (示例) ==========
if __name__ == "__main__":
    model = SurrogateBinarySNN_5ConvRes(time_steps=TIME_STEPS).to(device)
    criterion = nn.CrossEntropyLoss()

    # 如果想用学习率调度器，可加:
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 120, 160], gamma=0.2
    )

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, criterion, epoch)
        test_acc = test(model, test_loader, criterion, epoch)
        scheduler.step()  # 调度器更新

        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model_5conv_res_skip.pth")
            print(
                f"==> New best accuracy: {best_acc:.2f}% at epoch {best_epoch}, model saved!"
            )

    print(
        f"Training completed. Best test accuracy: {best_acc:.2f}% (epoch {best_epoch})."
    )
