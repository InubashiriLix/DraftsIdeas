import torch
import torch.nn as nn


class LIFNeuronSTDP(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super(LIFNeuronSTDP, self).__init__()
        self.threshold = threshold
        self.decay = decay

    def forward(self, input_current, mem):
        mem = self.decay * mem + input_current
        spike = (mem >= self.threshold).float()
        mem = mem * (1 - spike)
        return spike, mem


class STDP_SNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        time_steps=10,
        A_plus=0.01,
        A_minus=0.012,
        tau_pre=20.0,
        tau_post=20.0,
    ):
        super(STDP_SNN, self).__init__()
        self.time_steps = time_steps
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_pre = tau_pre
        self.tau_post = tau_post

        # NOTE: the full connection layer use the bias as 0
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

        self.lif_hidden = LIFNeuronSTDP(threshold=1.0, decay=0.9)
        self.lif_output = LIFNeuronSTDP(threshold=1.0, decay=0.9)

        # NOTE: update the trace of pre and post
        self.pre_trace_fc1 = torch.zeros(input_size)
        self.post_trace_fc1 = torch.zeros(hidden_size)
        self.pre_trace_fc2 = torch.zeros(hidden_size)
        self.post_trace_fc2 = torch.zeros(output_size)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # confirm the input on the correct device
        self.pre_trace_fc1 = self.pre_trace_fc1.to(device)
        self.post_trace_fc1 = self.post_trace_fc1.to(device)
        self.pre_trace_fc2 = self.pre_trace_fc2.to(device)
        self.post_trace_fc2 = self.post_trace_fc2.to(device)

        # initialize the membrane potential and output spike
        mem_hidden = torch.zeros(batch_size, self.fc1.out_features, device=device)
        mem_output = torch.zeros(batch_size, self.fc2.out_features, device=device)
        out_spike_sum = torch.zeros(batch_size, self.fc2.out_features, device=device)

        # enumerate the time steps to train
        for t in range(self.time_steps):
            # forward pass: fully connected -> LIF_hidden -> fully connected -> LIF_output
            current_hidden = self.fc1(x)
            spike_hidden, mem_hidden = self.lif_hidden(current_hidden, mem_hidden)

            current_output = self.fc2(spike_hidden)
            spike_output, mem_output = self.lif_output(current_output, mem_output)

            out_spike_sum += spike_output

            exp_factor_pre = torch.exp(torch.tensor(-1.0 / self.tau_pre, device=device))
            exp_factor_post = torch.exp(
                torch.tensor(-1.0 / self.tau_post, device=device)
            )

            # udpate trace of fullconnection layer 1
            pre_mean = x.mean(dim=0)  # linked to input layer
            post_mean_hidden = spike_hidden.mean(dim=0)  #

            self.pre_trace_fc1 = self.pre_trace_fc1 * exp_factor_pre + pre_mean
            self.post_trace_fc1 = (
                self.post_trace_fc1 * exp_factor_post + post_mean_hidden
            )

            # update the trace of fullconnection layer 2
            self.pre_trace_fc2 = (
                self.pre_trace_fc2 * exp_factor_pre + spike_hidden.mean(dim=0)
            )
            post_mean_output = spike_output.mean(dim=0)
            self.post_trace_fc2 = (
                self.post_trace_fc2 * exp_factor_post + post_mean_output
            )

            """
            calculate the delta weight of full connection layer
            delta_j_i = A_plus * post_j * pre_i - A_minus * post_trace_j * input_i
            then add the delta to the original weight
            fc_weight += delta_j_i(actually delta of full weight)
            """
            # NOTE: STDP update fc 1 (hidden_size, input_size)
            delta_w_fc1 = self.A_plus * torch.ger(
                spike_hidden.mean(dim=0), self.pre_trace_fc1
            ) - self.A_minus * torch.ger(self.post_trace_fc1, x.mean(dim=0))
            self.fc1.weight.data += delta_w_fc1

            # NOTE: STDP update fc 2 (hidden_size, input_size)
            delta_w_fc2 = self.A_plus * torch.ger(
                spike_output.mean(dim=0), self.pre_trace_fc2
            ) - self.A_minus * torch.ger(self.post_trace_fc2, spike_hidden.mean(dim=0))
            self.fc2.weight.data += delta_w_fc2

        return out_spike_sum


if __name__ == "__main__":
    model = STDP_SNN(input_size=5, hidden_size=10, output_size=2, time_steps=20)
    x = torch.randn(3, 5)  # batch_size = 3
    output = model(x)
    print("output integrated paluse", output)
