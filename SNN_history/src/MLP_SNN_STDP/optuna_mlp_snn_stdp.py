import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import optuna
from optuna.trial import Trial

# -----------------------------
# Original SNN + STDP code
#  (poisson_encode, LIFNeuron, SNN_MLP_STDP, stdp_update, train_stdp, test_stdp)
#  has been truncated for brevity, but is assumed to be available in the same script.
# -----------------------------

# Just for reference; keep them as you had:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms as before:
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
    train_dataset, batch_size=64, shuffle=True, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False, drop_last=False
)


# -----------------------------
# Objective function for Optuna
# -----------------------------
def objective(trial: Trial) -> float:
    """
    Define how to train and evaluate our SNN, returning a metric
    (e.g. test accuracy) that Optuna will try to maximize.
    """
    # Step 1: Suggest hyperparameters
    threshold = trial.suggest_float("threshold", 0.5, 1.0)  # e.g. [0.5,1.0]
    decay = trial.suggest_float("decay", 0.3, 0.99)  # e.g. [0.3,0.99]
    lr_stdp = trial.suggest_float("lr_stdp", 1e-5, 1e-3, log=True)
    A_pos = trial.suggest_float("A_pos", 1e-3, 1e-1, log=True)
    # For negative update, we'll tie A_neg = -A_pos (common approach)
    A_neg = -A_pos
    weight_clip_value = trial.suggest_float("weight_clip_value", 0.01, 0.5)
    hidden_size = trial.suggest_int("hidden_size", 200, 800, step=100)

    # You could tune time_window as well
    time_window = trial.suggest_int("time_window", 4, 10)

    # Optionally tune epochs (but usually you'd fix it)
    # num_epochs = trial.suggest_int("epochs", 5, 15)
    # Or keep it small for speed (like 5 epochs)
    num_epochs = 5

    # Step 2: Build your SNN model with these hyperparameters
    input_size = 32 * 32
    output_size = 10

    # We define the model as before, but with parameters from trial
    # NOTE: We must inject threshold, decay, etc. into LIFNeuron,
    #       or store them as global. Here's an easy way: create
    #       custom LIFNeuron with the trial's threshold/decay.
    class CustomLIFNeuron:
        def __init__(self, threshold, decay, soft_reset=True, device="cpu"):
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

    class SNN_MLP_STDP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, device="cpu"):
            super(SNN_MLP_STDP, self).__init__()
            self.device = device

            # Weight initialization
            self.W1 = nn.Parameter(0.01 * torch.randn(input_size, hidden_size))
            self.W2 = nn.Parameter(0.01 * torch.randn(hidden_size, output_size))

            self.W1.requires_grad_(False)
            self.W2.requires_grad_(False)

            self.lif1 = CustomLIFNeuron(threshold, decay, True, device)
            self.lif2 = CustomLIFNeuron(threshold, decay, True, device)

        def forward_once(self, input_spikes):
            hidden_u = torch.matmul(input_spikes, self.W1)
            hidden_spikes = self.lif1.forward(hidden_u)
            output_u = torch.matmul(hidden_spikes, self.W2)
            output_spikes = self.lif2.forward(output_u)
            return hidden_spikes, output_spikes

        def reset_neuron_state(self):
            self.lif1.reset_state()
            self.lif2.reset_state()

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

    # Re-define STDP update so we can incorporate (lr_stdp, A_pos, A_neg, weight_clip_value)
    def stdp_update(
        model: SNN_MLP_STDP,
        in_spikes_t: torch.Tensor,
        hid_spikes_t: torch.Tensor,
        out_spikes_t: torch.Tensor,
        labels: torch.Tensor,
    ):
        batch_size = labels.size(0)

        # label => one_hot
        label_one_hot = torch.zeros(batch_size, output_size, device=model.device)
        label_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # correct vs. wrong output spikes
        correct_mask = label_one_hot * out_spikes_t
        wrong_mask = (1.0 - label_one_hot) * out_spikes_t

        # (1) hidden->output
        pre_spike_h = hid_spikes_t.unsqueeze(2)
        post_spike_correct = correct_mask.unsqueeze(1)
        post_spike_wrong = wrong_mask.unsqueeze(1)

        dW2_pos = A_pos * torch.sum(pre_spike_h * post_spike_correct, dim=0)
        dW2_neg = A_neg * torch.sum(pre_spike_h * post_spike_wrong, dim=0)
        dW2 = dW2_pos + dW2_neg
        model.W2.data += lr_stdp * dW2

        # (2) input->hidden (reward-modulated STDP as example)
        total_correct_spikes = torch.sum(correct_mask, dim=1)
        total_wrong_spikes = torch.sum(wrong_mask, dim=1)
        global_reward = total_correct_spikes - total_wrong_spikes
        global_reward = global_reward.view(batch_size, 1, 1)

        pre_spike_in = in_spikes_t.unsqueeze(2)
        post_spike_h2 = hid_spikes_t.unsqueeze(1)
        sign_r = torch.sign(global_reward)
        dW1_batch = A_pos * (pre_spike_in * post_spike_h2) * sign_r
        dW1 = torch.sum(dW1_batch, dim=0)
        model.W1.data += lr_stdp * dW1

        # weight clipping
        model.W1.data.clamp_(-weight_clip_value, weight_clip_value)
        model.W2.data.clamp_(-weight_clip_value, weight_clip_value)

    def train_stdp(model, loader, time_window, epoch):
        model.train()
        total_samples = 0
        correct_count = 0
        for batch_idx, (data_, labels_) in enumerate(loader):
            data_ = data_.to(device)
            labels_ = labels_.to(device)
            bs_ = data_.size(0)
            total_samples += bs_

            # Poisson encode
            x_spikes = poisson_encode(data_, time_window)  # [T, B, 1, 32, 32]
            x_spikes_2d = x_spikes.view(time_window, bs_, -1)

            model.reset_neuron_state()
            out_spike_sum = torch.zeros(bs_, output_size, device=device)

            for t in range(time_window):
                hid_spikes_t, out_spikes_t = model.forward_once(x_spikes_2d[t])
                stdp_update(model, x_spikes_2d[t], hid_spikes_t, out_spikes_t, labels_)
                out_spike_sum += out_spikes_t

            pred_labels = torch.argmax(out_spike_sum, dim=1)
            correct_count += (pred_labels == labels_).sum().item()

        acc_ = 100.0 * correct_count / total_samples
        # Print or log
        # print(f"Epoch[{epoch}] - Train Accuracy = {acc_:.2f}%")
        return acc_

    def test_stdp(model, loader, time_window):
        model.eval()
        total_samples = 0
        correct_count = 0
        with torch.no_grad():
            for data_, labels_ in loader:
                data_ = data_.to(device)
                labels_ = labels_.to(device)
                bs_ = data_.size(0)
                total_samples += bs_

                x_spikes = poisson_encode(data_, time_window)
                x_spikes_2d = x_spikes.view(time_window, bs_, -1)

                model.reset_neuron_state()
                out_spike_sum = torch.zeros(bs_, output_size, device=device)
                for t in range(time_window):
                    _, out_spikes_t = model.forward_once(x_spikes_2d[t])
                    out_spike_sum += out_spikes_t

                pred = torch.argmax(out_spike_sum, dim=1)
                correct_count += (pred == labels_).sum().item()
        acc_ = 100.0 * correct_count / total_samples
        return acc_

    # Create the model
    snn_model = SNN_MLP_STDP(input_size, hidden_size, output_size, device=device)
    snn_model.to(device)

    # Training loop
    for ep in range(1, num_epochs + 1):
        # You can track the train accuracy, but let's not print too verbosely
        _ = train_stdp(snn_model, train_loader, time_window, ep)

    # Test after training
    test_accuracy = test_stdp(snn_model, test_loader, time_window)
    return test_accuracy


# -----------------------------
# Using Optuna to search
# -----------------------------
if __name__ == "__main__":
    # Create study for maximizing accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)  # NOTE: e.g. 10 trials; adjust as needed

    print("Best trial:")
    trial_ = study.best_trial
    print("  Value (Accuracy):", trial_.value)
    print("  Params :")
    for key, val in trial_.params.items():
        print(f"    {key}: {val}")
