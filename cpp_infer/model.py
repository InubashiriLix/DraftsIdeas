import torch
import torch.nn as nn
import torch.nn.functional as f


class ModelCnn(nn.Module):
    def __init__(self, input_x: int, input_y: int, num_classes: int):
        super(ModelCnn, self).__init__()

        self.input_x, self.input_y = input_x, input_y
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * (input_x // 4) * (input_y // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x
