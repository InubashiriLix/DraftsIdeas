import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()
        # (1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # (16, 28, 28)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 28 - 14
        # (16, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # (64, 14, 14)
        self.pool2 = nn.MaxPool2d(kernel_size=2) # 14 - 7
        # (64, 7, 7)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, stride=1, padding=1)
        # (10, 7, 7)
        self.fc = nn.Linear(in_features=10 * 7 * 7, out_features=10)


     def forward(self, x):
        x = self.pool1(self.conv1(x))
                x = self.pool1(self.conv1(x))
