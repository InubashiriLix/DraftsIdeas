import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# -----------------------------
#        超参数
# -----------------------------
batch_size = 64
num_epochs = 5  # 演示用，可加大
time_window = 8  # 脉冲时序长度
num_classes = 10
height, width = 32, 32

# 线性特征空间维度
feature_dim = 512  # 可调

# Linear特征映射初始化
# 是否随机固定映射：True => 不训练该层
fixed_feature_extractor = True

# LIF 超参数
threshold = 1.0
decay = 0.9
soft_reset = True

# STDP 超参数
lr_stdp = 0.05
A_pos = 0.01
A_neg = -0.01
weight_clip = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
#   数据集：CIFAR-10 (灰度)
# -----------------------------
transform_train = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)


# -----------------------------
#   Poisson编码
# -----------------------------
def poisson_encode(x, time_window):
    """
    x: [B, D], ∈ [0,1] => 发放率
    return: [time_window, B, D], 0/1
    """
    freq = torch.clamp(x, 0, 1)
    rand_vals = torch.rand((time_window,) + x.shape, device=x.device)
    spikes = (rand_vals < freq.unsqueeze(0)).float()
    return spikes


# -----------------------------
#   简易 LIF 神经元
# -----------------------------
class LIFNeuron:
    def __init__(self, threshold=1.0, decay=0.9, soft_reset=True, device="cpu"):
        self.threshold = threshold
        self.decay = decay
        self.soft_reset = soft_reset
        self.mem = None
        self.device = device

    def forward(self, input_):
        if self.mem is None or self.mem.shape != input_.shape:
            self.mem = torch.zeros_like(input_, device=self.device)
        self.mem = self.mem * self.decay + input_

        spike = (self.mem >= self.threshold).float()
        if self.soft_reset:
            self.mem = self.mem - spike * self.threshold
        else:
            self.mem = self.mem * (1 - spike)

        return spike

    def reset(self):
        self.mem = None


# -----------------------------
#   (1) 线性特征提取层
#       - 输入: [B,1,32,32] => flatten => [B, 1024]
#       - 输出: [B, feature_dim]
#       - 可选择固定随机，或BP训练
# -----------------------------
class LinearFeatureExtractor(nn.Module):
    def __init__(self, in_features=1024, out_features=256, fixed=True):
        super(LinearFeatureExtractor, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if fixed:
            # 随机初始化, 不参与训练
            self.linear.weight.data.normal_(0, 0.01)  # 可调
            self.linear.bias.data.zero_()
            self.linear.weight.requires_grad_(False)
            self.linear.bias.requires_grad_(False)

    def forward(self, x):
        """
        x: [B,1,32,32]
        return: [B, out_features], clamp到[0,1]
        """
        B = x.size(0)
        # flatten
        x_flat = x.view(B, -1)  # [B, 1024]
        feat = self.linear(x_flat)
        feat = nn.ReLU()(feat)
        # 这里让特征处于[0,1], 简易做法
        # 你也可改别的范围
        feat = torch.clamp(feat, 0.0, 1.0)
        return feat


# -----------------------------
#   (2) SNN 分类层: feature_dim -> 10
#       - 突触权重 W_snn 用 STDP 更新
# -----------------------------
class SNNClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, device="cpu"):
        super(SNNClassifier, self).__init__()
        # 突触
        self.W = nn.Parameter(0.01 * torch.randn(in_dim, out_dim), requires_grad=False)
        self.lif = LIFNeuron(threshold, decay, soft_reset, device=device)
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward_once(self, x_spike):
        """
        x_spike: [B, in_dim], 0/1
        return: out_spike: [B, out_dim]
        """
        I_out = torch.matmul(x_spike, self.W)  # [B, out_dim]
        out_spike = self.lif.forward(I_out)
        return out_spike

    def reset_state(self):
        self.lif.reset()


# -----------------------------
#   标签驱动 STDP
#   - 若正确神经元发放 => 正向更新
#   - 若错误神经元发放 => 负向更新
# -----------------------------
def stdp_update(W_snn, in_spike, out_spike, labels):
    """
    W_snn: [in_dim, out_dim]
    in_spike: [B, in_dim]
    out_spike: [B, out_dim]
    labels: [B]
    """
    B = labels.size(0)
    label_one_hot = torch.zeros(B, num_classes, device=labels.device)
    label_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    correct_mask = label_one_hot * out_spike  # [B, out_dim]
    wrong_mask = (1 - label_one_hot) * out_spike

    pre_spike = in_spike.unsqueeze(2)  # [B, in_dim, 1]
    post_correct = correct_mask.unsqueeze(1)  # [B, 1, out_dim]
    post_wrong = wrong_mask.unsqueeze(1)

    dW_pos = A_pos * torch.sum(pre_spike * post_correct, dim=0)  # [in_dim, out_dim]
    dW_neg = A_neg * torch.sum(pre_spike * post_wrong, dim=0)

    dW = dW_pos + dW_neg

    W_snn.data += lr_stdp * dW
    W_snn.data.clamp_(-weight_clip, weight_clip)


# -----------------------------
#   (3) 整合：FeatureExtractor + SNN
# -----------------------------
class LinearPlusSNN(nn.Module):
    def __init__(self, feature_dim, device="cpu"):
        super(LinearPlusSNN, self).__init__()
        # 线性特征提取
        self.feature_extractor = LinearFeatureExtractor(
            in_features=1024, out_features=feature_dim, fixed=fixed_feature_extractor
        )
        # SNN 分类层
        self.snn = SNNClassifier(feature_dim, num_classes, device=device)
        self.device = device

    def forward_feature(self, x):
        # 得到 [B, feature_dim], in [0,1]
        feat = self.feature_extractor(x)
        return feat

    def forward_snn_once(self, feat_spike):
        # 用SNN分类层
        out_spike = self.snn.forward_once(feat_spike)
        return out_spike

    def reset_snn_state(self):
        self.snn.reset_state()


# -----------------------------
#   训练
# -----------------------------
def train_one_epoch(model, train_loader, time_window, epoch):
    model.train()
    total_samples = 0
    correct = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        B = data.size(0)
        total_samples += B

        # 1) 提取特征 => [B, feature_dim]
        with torch.no_grad():  # 因为默认不训练特征提取层
            feat = model.forward_feature(data)

        # 2) Poisson 编码 => [time_window, B, feature_dim]
        feat_spike_seq = poisson_encode(feat, time_window)

        # 3) SNN 前向 & STDP
        model.reset_snn_state()
        out_spike_sum = torch.zeros(B, num_classes, device=device)
        for t in range(time_window):
            out_spike_t = model.forward_snn_once(feat_spike_seq[t])
            # STDP 更新
            stdp_update(model.snn.W, feat_spike_seq[t], out_spike_t, labels)
            # 统计脉冲
            out_spike_sum += out_spike_t

        # 4) 计算本batch准确率
        preds = out_spike_sum.argmax(dim=1)
        correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total_samples
    print(f"Epoch[{epoch}] Train Accuracy = {acc:.2f}%")


def test_snn(model, test_loader, time_window):
    model.eval()
    total_samples = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            B = data.size(0)
            total_samples += B

            feat = model.forward_feature(data)  # [B, feature_dim]
            feat_spike_seq = poisson_encode(feat, time_window)
            model.reset_snn_state()

            out_spike_sum = torch.zeros(B, num_classes, device=device)
            for t in range(time_window):
                out_spike_t = model.forward_snn_once(feat_spike_seq[t])
                out_spike_sum += out_spike_t

            preds = out_spike_sum.argmax(dim=1)
            correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total_samples
    print(f"Test Accuracy = {acc:.2f}%")
    return acc


# -----------------------------
#   主流程
# -----------------------------
if __name__ == "__main__":
    model = LinearPlusSNN(feature_dim, device=device).to(device)

    best_acc = 0.0
    for ep in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, time_window, ep)
        acc = test_snn(model, test_loader, time_window)
        if acc > best_acc:
            best_acc = acc

    print("Training finished. Best Test Accuracy= %.2f%%" % best_acc)
