import torch
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model.SNN_CLASIFICATION_IMAGE import STDP_ConvNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# prepare the transforms
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5,)),  # [-1 ~ 1]
    ]
)

# download and load the dataset
train_data = torchvision.datasets.CIFAR10(
    root="../data/cifar10",
    train=True,
    transform=transform,
    download=True,
)
test_data = torchvision.datasets.CIFAR10(
    root="../data/cifar10",
    train=False,
    transform=transform,
    download=True,
)


def binary_encode(tensor):
    # 输入形状应为 (batch, 1, 32, 32)
    tensor = (tensor + 1) / 2 * 255  # [-1,1] → [0,255]
    tensor = tensor.byte().to(torch.uint8)

    batch_size, channels, H, W = tensor.shape
    time_steps = 8

    # 创建编码张量 (batch, time, channels, H, W)
    encoded = torch.zeros(
        batch_size,
        time_steps,
        channels,
        H,
        W,
        dtype=torch.float32,
        device=tensor.device,
    )

    for t in range(time_steps):
        bit = (tensor >> (7 - t)) & 1  # 高位优先
        encoded[:, t] = bit.float()  # 保持通道维度

    return encoded  # 输出形状 (batch, 8, 1, 32, 32)


train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)


def train_with_accuracy(model, train_loader, device, epochs=10):
    model = model.to(device)

    for epoch in range(epochs):
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            # 数据预处理
            data = binary_encode(data).to(device)
            targets = targets.to(device)

            # 前向传播（自动更新权重）
            outputs = model(data)

            # 计算预测结果
            _, predicted = torch.max(outputs, 1)

            # 统计准确率
            correct = (predicted == targets).sum().item()
            total = targets.size(0)
            epoch_correct += correct
            epoch_total += total

            # 每10个batch打印一次进度
            if batch_idx % 10 == 0:
                batch_acc = correct / total
                print(f"Epoch {epoch + 1} | Batch {batch_idx} | Acc: {batch_acc:.2f}")

            batch_count += 1

        # 计算epoch准确率
        epoch_acc = epoch_correct / epoch_total
        print(f"Epoch {epoch + 1} Completed | Training Acc: {epoch_acc:.4f}")


if __name__ == "__main__":
    # model = STDP_ConvNet().to(DEVICE)
    # for data, _ in train_loader:
    #     data = binary_encode(data).to(device)
    #     outputs = model(data)
    #
    # print(f"Epoch ")
    train_with_accuracy(STDP_ConvNet(), train_loader, DEVICE, epochs=100)
