import torch
import torch.nn as nn

KERNEL_SIZE = 5


class LIFNeuronSTDP(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay

    def forward(self, inputs, mem):
        mem = self.decay * mem + inputs
        spike = (mem >= self.threshold).float()
        mem = mem * (1 - spike)
        return spike, mem


class STDP_ConvNet(nn.Module):
    def __init__(self, time_steps=8, A_plus=0.04, A_minus=0.03):
        super().__init__()
        self.time_steps = time_steps

        # 卷积层
        self.conv1 = nn.Conv2d(1, 12, 5, padding=2, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5, padding=2, bias=False)

        # 全连接层
        self.fc = nn.Linear(32 * 8 * 8, 10, bias=False)

        # LIF神经元
        self.lif1 = LIFNeuronSTDP(threshold=0.5, decay=0.85)
        self.lif2 = LIFNeuronSTDP(threshold=0.5, decay=0.85)
        self.lif3 = LIFNeuronSTDP(threshold=0.5, decay=0.85)

        # STDP参数
        self.A_plus = A_plus
        self.A_minus = A_minus

        # 初始化迹变量
        self.register_buffer("pre_trace_conv1", torch.zeros(1))  # 输入通道数1
        self.register_buffer("post_trace_conv1", torch.zeros(12))  # 输出通道数12
        self.register_buffer("pre_trace_conv2", torch.zeros(12))  # 输入通道数12
        self.register_buffer("post_trace_conv2", torch.zeros(32))  # 输出通道数32
        self.register_buffer("pre_trace_fc", torch.zeros(32 * 8 * 8))
        self.register_buffer("post_trace_fc", torch.zeros(10))

    def stdp_update(self, pre, post, layer):
        if isinstance(layer, nn.Conv2d):
            out_ch, in_ch, k_h, k_w = layer.weight.shape

            # 扩展维度以支持广播
            post = post.view(out_ch, 1, 1, 1)  # [12] → [12,1,1,1]
            pre = pre.view(1, in_ch, 1, 1)  # [1] → [1,1,1,1]

            # 计算更新量
            delta_w = self.A_plus * post * pre
            delta_w -= self.A_minus * pre * post

            # 扩展为卷积核形状
            delta_w = delta_w.expand(out_ch, in_ch, k_h, k_w)
        elif isinstance(layer, nn.Linear):
            # LTP：外积 post × pre，得到 [out_features, in_features]
            ltp = self.A_plus * torch.outer(post, pre)

            # LTD：也用同样的形状 (post × pre)
            ltd = self.A_minus * torch.outer(post, pre)

            # 两者相减，仍是 [out_features, in_features]
            delta_w = ltp - ltd

            # 如果 delta_w 和 layer.weight.shape 一致，就不需要再 view()
            # 如果你想写得更简洁，可以直接：
            # delta_w = self.A_plus * torch.outer(post, pre)
            # delta_w -= self.A_minus * torch.outer(post, pre)

            layer.weight.data += delta_w

        layer.weight.data += delta_w

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # 初始化膜电位
        mem_conv1 = torch.zeros(batch_size, 12, 32, 32, device=device)
        mem_conv2 = torch.zeros(batch_size, 32, 16, 16, device=device)
        mem_fc = torch.zeros(batch_size, 10, device=device)

        # 脉冲计数
        total_spikes = torch.zeros(batch_size, 10, device=device)

        for t in range(self.time_steps):
            input_spike = x[:, t]  # [batch,1,32,32]

            # 第一卷积层
            conv1_out = self.conv1(input_spike)
            spike1, mem_conv1 = self.lif1(conv1_out, mem_conv1)
            pooled = self.pool(spike1)  # [batch,12,16,16]

            # 第二卷积层
            conv2_out = self.conv2(pooled)
            spike2, mem_conv2 = self.lif2(conv2_out, mem_conv2)
            pooled2 = self.pool(spike2)  # [batch,32,8,8]

            # 全连接层
            flattened = pooled2.view(batch_size, -1)
            fc_out = self.fc(flattened)
            spike3, mem_fc = self.lif3(fc_out, mem_fc)

            # 更新迹变量
            self.pre_trace_conv1 = 0.9 * self.pre_trace_conv1 + input_spike.mean()
            self.post_trace_conv1 = 0.9 * self.post_trace_conv1 + spike1.mean(
                dim=(0, 2, 3)
            )
            self.pre_trace_conv2 = 0.9 * self.pre_trace_conv2 + pooled.mean(
                dim=(0, 2, 3)
            )
            self.post_trace_conv2 = 0.9 * self.post_trace_conv2 + spike2.mean(
                dim=(0, 2, 3)
            )
            self.pre_trace_fc = 0.9 * self.pre_trace_fc + flattened.mean(dim=0)
            self.post_trace_fc = 0.9 * self.post_trace_fc + spike3.mean(dim=0)

            # STDP更新
            self.stdp_update(self.pre_trace_conv1, self.post_trace_conv1, self.conv1)
            self.stdp_update(self.pre_trace_conv2, self.post_trace_conv2, self.conv2)
            self.stdp_update(self.pre_trace_fc, self.post_trace_fc, self.fc)

            total_spikes += spike3

        return total_spikes
