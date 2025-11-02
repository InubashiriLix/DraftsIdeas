import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.datasets
from torchvision import transforms

# download the mnist dataset
mnist_train = torchvision.datasets.MNIST(
    root="../data/mnist/", train=True, transform=transforms.ToTensor(), download=True
)
mnist_test = torchvision.datasets.MNIST(
    root="../data/mnist/", train=True, transform=transforms.ToTensor(), download=True
)

print(mnist_train[1][0].shape)
