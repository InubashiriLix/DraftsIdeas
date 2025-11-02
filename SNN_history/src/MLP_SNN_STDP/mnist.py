import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import optuna

# ======================================
#  全局: 数据加载 (MNIST)
# ======================================
batch_size = 64
input_size = 28 * 28  # MNIST 28x28
output_size = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# MNIST 数据集
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # [0,1]
    ]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)


# ======================================
#  Poisson 编码
# ======================================
def poisson_encode(x, time_window):
    """
    x: [B,1,28,28], 值 ∈ [0,1]
    返回: [time_window, B, 28*28]
    """
    freq = torch.clamp(x, 0, 1)
    rand_vals = torch.rand((time_window,) + x.shape, device=x.device)
    spikes = (rand_vals < freq.unsqueeze(0)).float()
    return spikes.view(time_window, x.size(0), -1)


# ======================================
#  LIF 神经元 (无BP, 简易实现)
# ======================================
class LIFNeuron:
    def __init__(self, threshold=1.0, decay=0.9, soft_reset=True, device="cpu"):
        self.threshold = threshold
        self.decay = decay
        self.soft_reset = soft_reset
        self.mem = None
        self.device = device

    def forward(self, input_current):
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


# ======================================
#  单层 SNN: input_size -> output_size
# ======================================
class SingleLayerSNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        threshold,
        decay,
        soft_reset,
        init_wscale=0.01,
        device="cpu",
    ):
        super(SingleLayerSNN, self).__init__()
        self.device = device

        # W: [input_size, output_size]
        self.W = nn.Parameter(init_wscale * torch.randn(input_size, output_size))
        self.W.requires_grad_(False)  # 不走BP

        # 输出层 LIF
        self.lif = LIFNeuron(threshold, decay, soft_reset, device=device)

    def forward_once(self, input_spikes):
        """
        input_spikes: [B, input_size], 0/1
        返回: [B, output_size], 0/1
        """
        I_out = torch.matmul(input_spikes, self.W)  # [B, output_size]
        out_spike = self.lif.forward(I_out)
        return out_spike

    def reset_neuron_state(self):
        self.lif.reset_state()


# ======================================
#  标签驱动 STDP 更新
# ======================================
def stdp_update(
    model,
    in_spikes_t,
    out_spikes_t,
    labels,
    A_positive,
    A_negative,
    lr_stdp,
    weight_clip,
):
    """
    model.W: [input_size, output_size]
    in_spikes_t:  [B, input_size]
    out_spikes_t: [B, output_size]
    labels:       [B]
    """
    batch_size = labels.size(0)
    label_one_hot = torch.zeros(batch_size, output_size, device=labels.device)
    label_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    correct_mask = label_one_hot * out_spikes_t  # [B, output_size]
    wrong_mask = (1.0 - label_one_hot) * out_spikes_t

    pre_spike = in_spikes_t.unsqueeze(2)  # [B, input_size, 1]
    post_spike_correct = correct_mask.unsqueeze(1)  # [B, 1, output_size]
    post_spike_wrong = wrong_mask.unsqueeze(1)

    dW_pos = A_positive * torch.sum(pre_spike * post_spike_correct, dim=0)
    dW_neg = A_negative * torch.sum(pre_spike * post_spike_wrong, dim=0)
    dW = dW_pos + dW_neg

    with torch.no_grad():
        model.W.data += lr_stdp * dW
        model.W.data.clamp_(-weight_clip, weight_clip)


# ======================================
#  训练、测试过程
# ======================================
def train_stdp(
    model,
    train_loader,
    time_window,
    A_positive,
    A_negative,
    lr_stdp,
    weight_clip,
    epochs=5,
):
    """
    训练若干 epoch，返回最后一次 epoch 的 (train_acc, test_acc)
    为了加速搜索，可以减少 epochs
    """
    for epoch in range(1, epochs + 1):
        model.train()
        total_samples = 0
        correct_count = 0
        for data, labels in train_loader:
            data, labels = data.to(model.device), labels.to(model.device)
            B = data.size(0)
            total_samples += B

            # Poisson 编码
            in_spike_seq = poisson_encode(data, time_window)  # [time_window, B, 784]

            # 重置膜电位
            model.reset_neuron_state()

            out_spike_sum = torch.zeros(B, output_size, device=model.device)

            for t in range(time_window):
                out_spikes_t = model.forward_once(in_spike_seq[t])
                # STDP 更新
                stdp_update(
                    model,
                    in_spike_seq[t],
                    out_spikes_t,
                    labels,
                    A_positive,
                    A_negative,
                    lr_stdp,
                    weight_clip,
                )
                # 累加脉冲
                out_spike_sum += out_spikes_t

            # 统计准确率
            preds = out_spike_sum.argmax(dim=1)
            correct_count += (preds == labels).sum().item()

        train_acc = 100.0 * correct_count / total_samples
        # 也可每个epoch后做一次 test，这里只做最后 epoch 的 test
        # print(f"Epoch[{epoch}] TrainAcc={train_acc:.2f}%")

    return train_acc


def test_stdp(model, test_loader, time_window):
    model.eval()
    total_samples = 0
    correct_count = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(model.device), labels.to(model.device)
            B = data.size(0)
            total_samples += B

            in_spike_seq = poisson_encode(data, time_window)
            model.reset_neuron_state()
            out_spike_sum = torch.zeros(B, output_size, device=model.device)

            for t in range(time_window):
                out_spikes_t = model.forward_once(in_spike_seq[t])
                out_spike_sum += out_spikes_t

            preds = out_spike_sum.argmax(dim=1)
            correct_count += (preds == labels).sum().item()

    acc = 100.0 * correct_count / total_samples
    return acc


# ======================================
#   Optuna 搜索的目标函数
# ======================================
def objective(trial):
    """
    在这里定义要搜索的超参数范围，以及如何训练和返回分数
    """
    # ---- 1) 采样超参数 ----
    time_window = trial.suggest_int("time_window", 5, 20)
    lr_stdp = trial.suggest_float("lr_stdp", 1e-5, 1e-1, log=True)
    A_positive = trial.suggest_float("A_positive", 0.01, 0.5)  # [0.01, 0.5]
    A_negative = -trial.suggest_float("A_negative_pos", 0.01, 0.5)  # 取负号
    weight_clip = trial.suggest_float("weight_clip", 0.05, 0.5)
    # 也可以把 LIF 的 threshold, decay 等一起加入搜索
    threshold = trial.suggest_float("threshold", 0.5, 2.0)
    decay = trial.suggest_float("decay", 0.8, 0.99)

    # ---- 2) 创建模型 ----
    model = SingleLayerSNN(
        input_size=input_size,
        output_size=output_size,
        threshold=threshold,
        decay=decay,
        soft_reset=True,
        init_wscale=0.01,
        device=device,
    ).to(device)

    # ---- 3) 训练 (为加速搜索, epochs可设小一些) ----
    # 注意: 如果 epochs 过大, 超参数搜索会很慢
    epochs = 3
    train_acc = train_stdp(
        model,
        train_loader,
        time_window,
        A_positive,
        A_negative,
        lr_stdp,
        weight_clip,
        epochs=epochs,
    )
    # ---- 4) 测试 ----
    test_acc = test_stdp(model, test_loader, time_window)

    # Trial记录一下可视化
    trial.report(test_acc, step=epochs)

    # 如果想做early stopping:
    # if trial.should_prune():
    #     raise optuna.TrialPruned()

    # 以测试集准确率 作为搜索目标 (最大化)
    return test_acc


# ======================================
#   主流程：Optuna 搜索
# ======================================
if __name__ == "__main__":
    import optuna

    # 创建study, 指定方向maximize
    study = optuna.create_study(direction="maximize")
    # 开始搜索, n_trials自己定, 例如 20
    study.optimize(objective, n_trials=100)

    print("==== Best Trial ====")
    best_trial = study.best_trial
    print(f"  Value (Test Acc): {best_trial.value}")
    print("  Params: ")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")

    # 如果想用best params再训一次可以:
    # best_params = best_trial.params
    # print("Use best_params to retrain the model in a separate step if you want...")

