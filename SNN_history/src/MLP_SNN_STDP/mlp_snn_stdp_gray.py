import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# -----------------------------
#        超参数
# -----------------------------
batch_size = 64
num_epochs = 20  # 演示用，可加大
time_window = 8  # 脉冲时序长度
height, width = 32, 32  # CIFAR-10原始大小
num_classes = 10

# LIF 超参数
threshold = 0.9
decay = 0.6
soft_reset = True

# STDP 相关超参数 (学习率等)
# 可根据实际实验情况多次尝试，可能需要很小很小，或配合归一化、裁剪等。
lr_stdp = 0.0001
A_pos = 0.01
A_neg = -0.01

# 为防止训练时权重爆炸/崩溃，可在每次更新后进行裁剪
weight_clip_value = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
#   数据集: 加载 & 预处理
#   - 转为灰度
#   - 进行数据增强 (train) 或基本转换 (test)
#   - 通常还会做归一化，这里可省略/或视情况而定
# -----------------------------
transform_train = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # 转为灰度
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
#   - 将连续像素转为 time_window 个脉冲序列
# -----------------------------
def poisson_encode(x, time_window):
    """
    x: [B,1,H,W], ∈ [0,1] 的灰度
    return: [time_window, B, 1, H, W]，0/1脉冲
    """
    freq = torch.clamp(x, 0, 1)  # 保证在[0,1]
    # 随机矩阵
    rand_vals = torch.rand((time_window,) + x.shape, device=x.device)
    spikes = (rand_vals < freq.unsqueeze(0)).float()
    return spikes


# -----------------------------
#   简易 LIF 神经元
#   - 只做膜电位+阈值放电
#   - 不参与自动求导
# -----------------------------
class LIFNeuron:
    def __init__(self, threshold=1.0, decay=0.9, soft_reset=True, device="cpu"):
        self.threshold = threshold
        self.decay = decay
        self.soft_reset = soft_reset
        self.mem = None
        self.device = device

    def forward(self, input_current):
        """
        input_current: [B, N_pre]
        return: spikes: [B, N_pre] (同样大小，因为这里是一对一单层处理)
        """
        if self.mem is None or self.mem.shape != input_current.shape:
            self.mem = torch.zeros_like(input_current, device=self.device)

        self.mem = self.mem * self.decay + input_current

        spikes = (self.mem >= self.threshold).float()
        if self.soft_reset:
            self.mem = self.mem - spikes * self.threshold
        else:
            self.mem = self.mem * (1 - spikes)

        return spikes

    def reset_state(self):
        self.mem = None


# -----------------------------
#   2层 MLP (input->hidden->output)
#   - 每层都对应 LIF
#   - 权重手动更新，不用optimizer
# -----------------------------
class SNN_MLP_STDP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device="cpu"):
        super(SNN_MLP_STDP, self).__init__()
        self.device = device

        # 初始化权重
        #   W1: [input_size, hidden_size]
        #   W2: [hidden_size, output_size]
        #   这里可以使用较小随机初始，避免膜电位过早饱和
        self.W1 = nn.Parameter(0.01 * torch.randn(input_size, hidden_size))
        self.W2 = nn.Parameter(0.01 * torch.randn(hidden_size, output_size))

        # 将其requires_grad设为False，表示我们不走传统BP
        self.W1.requires_grad_(False)
        self.W2.requires_grad_(False)

        # 两层LIF
        self.lif1 = LIFNeuron(threshold, decay, soft_reset, device)
        self.lif2 = LIFNeuron(threshold, decay, soft_reset, device)

    def forward_once(self, input_spikes):
        """
        针对单个时间步进行一次前向传播:
        input_spikes: [B, input_size]
        return:
          hidden_spikes: [B, hidden_size]
          output_spikes: [B, output_size]
        """
        # input -> hidden
        hidden_u = torch.matmul(input_spikes, self.W1)  # [B, hidden_size]
        hidden_spikes = self.lif1.forward(hidden_u)

        # hidden -> output
        output_u = torch.matmul(hidden_spikes, self.W2)  # [B, output_size]
        output_spikes = self.lif2.forward(output_u)

        return hidden_spikes, output_spikes

    def forward_time(self, input_spikes_time):
        """
        针对一个样本在 time_window 个时刻的脉冲
        input_spikes_time: [time_window, B, input_size]
        return:
          hidden_spikes_all: [time_window, B, hidden_size]
          output_spikes_all: [time_window, B, output_size]
        """
        self.reset_neuron_state()

        hidden_list = []
        output_list = []
        for t in range(input_spikes_time.shape[0]):
            hs, os = self.forward_once(input_spikes_time[t])
            hidden_list.append(hs)
            output_list.append(os)

        hidden_spikes_all = torch.stack(hidden_list, dim=0)  # [T, B, hidden_size]
        output_spikes_all = torch.stack(output_list, dim=0)  # [T, B, output_size]
        return hidden_spikes_all, output_spikes_all

    def reset_neuron_state(self):
        self.lif1.reset_state()
        self.lif2.reset_state()


# -----------------------------
#   标签驱动 STDP 更新
#   - 当“正确输出神经元”发放脉冲 => 正向更新
#   - 当“错误输出神经元”发放脉冲 => 负向更新
#   - 对 W1、W2 都做 STDP
#   - W1 的 STDP 则将 (input_spike, hidden_spike) 做一次 pre-post 规则
# -----------------------------
def stdp_update(
    model: SNN_MLP_STDP,
    in_spikes_t: torch.Tensor,
    hid_spikes_t: torch.Tensor,
    out_spikes_t: torch.Tensor,
    labels: torch.Tensor,
):
    """
    in_spikes_t:  [B, input_size]
    hid_spikes_t: [B, hidden_size]
    out_spikes_t: [B, output_size]
    labels:        [B]
    """
    batch_size = labels.size(0)

    # 先做输出层（hidden -> output）的 STDP
    # --------------------------------------------------
    # out_spikes_t in {0,1}, shape=[B, output_size]
    # labels => one_hot: shape=[B, output_size]
    label_one_hot = torch.zeros(batch_size, num_classes, device=model.device)
    label_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    # 正确神经元的脉冲 => 正向更新
    correct_mask = label_one_hot * out_spikes_t  # [B, output_size]
    # 错误神经元的脉冲 => 负向更新
    wrong_mask = (1.0 - label_one_hot) * out_spikes_t

    # pre_spike: hidden_spikes_t: [B, hidden_size]
    # post_spike_correct: correct_mask: [B, output_size]
    # => ΔW2 = A_pos * sum( pre_spike[:,i] * post_spike_correct[:,j] ) - A_neg * ...
    pre_spike_h = hid_spikes_t.unsqueeze(2)  # [B, hidden_size, 1]
    post_spike_correct = correct_mask.unsqueeze(1)  # [B, 1, output_size]
    post_spike_wrong = wrong_mask.unsqueeze(1)  # [B, 1, output_size]

    # 正向
    dW2_pos = A_pos * torch.sum(
        pre_spike_h * post_spike_correct, dim=0
    )  # [hidden_size, output_size]
    # 负向
    dW2_neg = A_neg * torch.sum(
        pre_spike_h * post_spike_wrong, dim=0
    )  # [hidden_size, output_size]
    dW2 = dW2_pos + dW2_neg

    # 更新
    model.W2.data += lr_stdp * dW2

    # 再做输入层（input -> hidden）的 STDP
    # --------------------------------------------------
    # 这里很多实现里会是无监督 STDP，即隐藏层只要发放就正向更新
    # 但也有人会把输出层的正确/错误信息传回来，对隐藏层的突触做奖惩（Reward-Modulated STDP）
    # 这里做个简单“全局奖励”思路：如果在这一时刻，有正确输出神经元发放，则对当前隐藏层突触正向更新，
    # 若只有错误神经元发放，则负向更新。若都没发放或发放相同，则几乎不更新（很粗糙）。
    # 或者你也可以只做无监督STDP。
    # 这里给你一个示例：基于 out_spikes_t.sum(1) 在正确neuron或错误neuron的差值进行一个全局奖惩。

    # （1）先计算：正向输出脉冲数量 total_correct_spikes, 错误输出脉冲数量 total_wrong_spikes
    #   shape=[B]
    total_correct_spikes = torch.sum(correct_mask, dim=1)  # [B]
    total_wrong_spikes = torch.sum(wrong_mask, dim=1)  # [B]

    # 我们定一个“全局reward信号”: r = total_correct_spikes - total_wrong_spikes
    #   r>0 => 正向
    #   r<0 => 负向
    #   r=0 => 不更新
    # 这只是一个示例，实际上可以更精细地区分
    global_reward = total_correct_spikes - total_wrong_spikes  # [B]
    # shape=[B,1,1], 用于广播
    global_reward = global_reward.view(batch_size, 1, 1)

    # pre_spike_in = in_spikes_t: [B, input_size]
    # post_spike_h = hid_spikes_t: [B, hidden_size]
    pre_spike_in = in_spikes_t.unsqueeze(2)  # [B, input_size, 1]
    post_spike_h = hid_spikes_t.unsqueeze(1)  # [B, 1, hidden_size]

    # 直接来一个： dW1_batch = A_pos * pre_spike_in * post_spike_h * sign(global_reward)
    #           sign>0 正向, sign<0 负向
    # 也可以更精细： r>0 的时候 A_pos, r<0 的时候 A_neg
    sign_r = torch.sign(global_reward)  # +1, 0, -1
    # 这里相当于 batch 内每条样本都会贡献一个突触更改，最后 sum 起来
    dW1_batch = A_pos * (pre_spike_in * post_spike_h) * sign_r

    # 在 batch 维度上求和 => [input_size, hidden_size]
    dW1 = torch.sum(dW1_batch, dim=0)
    model.W1.data += lr_stdp * dW1

    # 最后，对 W1、W2 做裁剪，防止数值爆炸
    model.W1.data.clamp_(-weight_clip_value, weight_clip_value)
    model.W2.data.clamp_(-weight_clip_value, weight_clip_value)


# -----------------------------
#   训练（STDP）
# -----------------------------
def train_stdp(model, train_loader, time_window, epoch):
    model.train()  # 虽然没用BP，这里仅表示处于训练阶段
    total_samples = 0
    correct_count = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)  # [B,1,32,32]
        labels = labels.to(device)
        batch_size_ = data.size(0)
        total_samples += batch_size_

        # Poisson编码
        x_spikes = poisson_encode(data, time_window)  # [time_window, B, 1, 32, 32]
        # 展平
        x_spikes_2d = x_spikes.view(time_window, batch_size_, -1)  # [T, B, input_size]

        # 重置膜电位
        model.reset_neuron_state()

        # 统计输出脉冲 (for 计算训练中的准确率)
        out_spike_sum = torch.zeros(batch_size_, num_classes, device=device)

        # time_window循环
        for t in range(time_window):
            # forward
            hid_spikes_t, out_spikes_t = model.forward_once(x_spikes_2d[t])
            # STDP 更新
            stdp_update(model, x_spikes_2d[t], hid_spikes_t, out_spikes_t, labels)

            # 用于统计准确率
            out_spike_sum += out_spikes_t

        # 选脉冲最多的类
        pred_labels = torch.argmax(out_spike_sum, dim=1)
        correct_count += (pred_labels == labels).sum().item()

    acc = 100.0 * correct_count / total_samples
    print(f"Epoch[{epoch}] Train Accuracy = {acc:.2f}%")


# -----------------------------
#   测试（固定权重，前向+投票）
# -----------------------------
def test_stdp(model, test_loader, time_window):
    model.eval()
    total_samples = 0
    correct_count = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)  # [B,1,32,32]
            labels = labels.to(device)
            batch_size_ = data.size(0)
            total_samples += batch_size_

            # Poisson编码
            x_spikes = poisson_encode(data, time_window)
            x_spikes_2d = x_spikes.view(time_window, batch_size_, -1)

            model.reset_neuron_state()
            out_spike_sum = torch.zeros(batch_size_, num_classes, device=device)

            for t in range(time_window):
                _, out_spikes_t = model.forward_once(x_spikes_2d[t])
                out_spike_sum += out_spikes_t

            pred = torch.argmax(out_spike_sum, dim=1)
            correct_count += (pred == labels).sum().item()

    acc = 100.0 * correct_count / total_samples
    print(f"Test Accuracy = {acc:.2f}%")
    return acc


# -----------------------------
#   主流程
# -----------------------------
if __name__ == "__main__":
    input_size = height * width  # 32*32=1024
    hidden_size = 500  # 可调，越大越慢，但也可能稍提高学习能力
    output_size = num_classes

    # 建立模型
    model = SNN_MLP_STDP(input_size, hidden_size, output_size, device=device)
    model.to(device)

    best_acc = 0.0
    for ep in range(1, num_epochs + 1):
        train_stdp(model, train_loader, time_window, ep)
        acc = test_stdp(model, test_loader, time_window)
        if acc > best_acc:
            best_acc = acc
            # 可以根据需要保存
            # torch.save(model.state_dict(), 'best_stdp_snn.pth')

    print("Training finished. Best Test Acc= %.2f%%" % best_acc)

