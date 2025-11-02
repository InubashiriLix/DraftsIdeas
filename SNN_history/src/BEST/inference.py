import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cifar10_conv4_linear2_better_lif import (
    SNN_CNN,
)  # 确保 model.py 里有 SNN_CNN 这个类

# ==============================
#  参数配置
# ==============================
batch_size = 64
time_window = 10  # SNN 的时间步
model_path = "better_model_conv4_linear2_better_lif.pth"  # 你的模型文件路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
#  数据集加载 (CIFAR-10)
# ==============================
test_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

test_dataset = datasets.CIFAR10(
    root="data", train=False, download=True, transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==============================
#  加载模型
# ==============================
model = SNN_CNN().to(device)  # 确保模型结构和训练时一致
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# ==============================
#  Inference 函数
# ==============================
def inference(model, device, test_loader):
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, time_window=time_window)
            pred = output.argmax(dim=1)
            total_samples += target.size(0)
            correct += pred.eq(target).sum().item()

    accuracy = 100.0 * correct / total_samples
    print(f"Inference Accuracy on CIFAR-10 test set: {accuracy:.2f}%")
    return accuracy


# ==============================
#  运行推理
# ==============================
if __name__ == "__main__":
    inference(model, device, test_loader)
