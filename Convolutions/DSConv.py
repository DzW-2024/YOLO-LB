import torch.nn as nn
import torch


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.dconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            groups=in_channels, # When groups=in_channel, it indicates performing per-channel convolution.
        )
        self.pconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)
