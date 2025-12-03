import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, reduction=2, cheap_op=nn.Conv2d, cheap_kernel_size=None):
        super().__init__()
        self.out_channels = out_channels
        self.reduction = reduction
        self.primary_channels = out_channels // reduction
        assert self.primary_channels * reduction == out_channels, \
            f"out_channels must be divisible by reduction! Current out_channels={out_channels}, reduction={reduction}"

        self.primary_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.primary_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )

        # Cheap kernel size, defaults to match the core convolution size, customizable via cheap_kernel_size
        self.cheap_kernel_size = cheap_kernel_size if cheap_kernel_size is not None else kernel_size
        self.cheap_padding = self.cheap_kernel_size // 2

        # Cheap transformation, use a uniformly passed kernel_size to avoid redundant assignments.
        self.cheap_op = cheap_op(
            in_channels=self.primary_channels,
            out_channels=(reduction - 1) * self.primary_channels,
            kernel_size=self.cheap_kernel_size,
            stride=1,
            padding=self.cheap_padding,
            dilation=dilation,
            groups=self.primary_channels,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        primary = self.primary_conv(x)
        ghost = self.cheap_op(primary)
        out = torch.cat([primary, ghost], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out

