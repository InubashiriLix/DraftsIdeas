import torch
import torch.nn as nn

class SpikingNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        """
        脉冲神经元（LIF 模型）
        :param threshold: 膜电位发放脉冲的阈值
        :param decay: 膜电位衰减因子（0~1之间）
        """
        super(SpikingNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay

    def forward(self, input_current, mem):
        """
        :param input_current: 当前时间步的输入电流
        :param mem: 上一时间步的膜电位
        :return:
            spike: 脉冲输出（0或1）
            mem: 更新后的膜电位（在发放脉冲的位置重置为0）
        """
        # 更新膜电位：先衰减再加上输入电流
        mem = self.decay * mem + input_current
        # 当膜电位大于等于阈值时，神经元发放脉冲（1），否则为0
        spike = (mem >= self.threshold).float()
        # 在发放脉冲的位置重置膜电位（可选择不同的重置策略）
        mem = mem * (1 - spike)
        return spike, mem


class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_steps=10):
        """
        一个简单的脉冲神经网络（SNN），包含两层全连接及脉冲神经元
        :param input_size: 输入特征数
        :param hidden_size: 隐藏层神经元数量
        :param output_size: 输出神经元数量
        :param time_steps: 模拟的时间步数
        """
        super(SNN, self).__init__()
        self.time_steps = time_steps

        # 第一层：输入 -> 隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 第二层：隐藏层 -> 输出层
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 为两层各自配备脉冲神经元模块
        self.neuron1 = SpikingNeuron(threshold=1.0, decay=0.9)
        self.neuron2 = SpikingNeuron(threshold=1.0, decay=0.9)

    def forward(self, x):
        batch_size = x.size(0)
        # 初始化每层的膜电位
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        
        # 用于累计输出脉冲（可看作网络在多个时间步上的响应）
        out_spike_sum = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        
        # 模拟多个时间步的动态演化
        for t in range(self.time_steps):
            # 第一层计算：全连接 + 脉冲神经元
            current1 = self.fc1(x)
            spike1, mem1 = self.neuron1(current1, mem1)
            
            # 第二层计算：全连接 + 脉冲神经元
            current2 = self.fc2(spike1)
            spike2, mem2 = self.neuron2(current2, mem2)
            
            # 累计输出层的脉冲
            out_spike_sum += spike2
        
        # 返回累计的脉冲数作为最终输出
        return out_spike_sum


# 示例：测试 SNN 网络的前向传播
if __name__ == "__main__":
    # 定义一个 SNN 模型：输入维度 2，隐藏层 10 个神经元，输出维度 1，模拟 20 个时间步
    model = SNN(input_size=2, hidden_size=10, output_size=1, time_steps=20)
    
    # 构造一个随机输入（batch 大小为 5）
    x = torch.rand(5, 2)
    
    # 前向传播得到输出脉冲累计
    output = model(x)
    print("输出脉冲累计：", output)

