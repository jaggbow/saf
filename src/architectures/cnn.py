import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        channels=[32, 64],
        kernel_sizes=[4, 3],
        strides=[2, 2],
        hidden_layer=512,
        out_size=64,):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_sizes[0], strides[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_sizes[1], strides[1])
        self.linear1 = nn.Linear(2304, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, out_size)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs / 255.))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x
