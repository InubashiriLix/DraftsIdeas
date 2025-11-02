import torch
import torch.nn as nn
import torch.utils.data as data


class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super(LIFNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay

    def forward(self, input_current, mem):
        mem = self.decay * mem + input_current
        spike = (mem >= self.threshold).float()
        # WARNING: simply reset the spike to zero, specific reset strategy can be used
        mem = mem * (1.0 - spike)
        return spike, mem


class SNN_LIF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_step=10):
        super(SNN_LIF, self).__init__()
        self.time_step = time_step

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.lif1 = LIFNeuron(threshold=1.0, decay=0.9)
        self.lif2 = LIFNeuron(threshold=1.0, decay=0.9)

    def forward(self, x):
        batch_size = x.size(0)
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=x.device)

        out_spikes = torch.zeros(batch_size, self.fc2.out_features, device=x.device)

        for t in range(self.time_step):
            current1 = self.fc1(x)
            spike1, mem1 = self.lif1(current1, mem1)
            current2 = self.fc2(spike1)
            spike2, mem2 = self.lif2(current2, mem2)

            return out_spikes


if __name__ == "__main__":
    model = SNN_LIF(input_size=5, hidden_size=10, output_size=2, time_step=20)
    x = torch.randn(3, 5)  # batch_size = 3
    output = model(x)
    print("output integrated paluse", output)
