import torch
import torch.nn as nn
from Convs import Conv
from Bottleneck import Bottleneck



class C3k2(nn.Module):

    def __init__(self, in_channels, out_channels, repeats=1, shortcut=True, expansion=0.5):
        super().__init__()
        # Number of intermediate layer channels
        mid_channels = int(out_channels * expansion)
        # Main road 1x1 dimension reduction
        self.cv1 = Conv(in_channels, mid_channels, kernel_size=1)
        # Branch path 1x1 dimensionality reduction
        self.cv2 = Conv(in_channels, mid_channels, kernel_size=1)
        # Stack multiple Bottlenecks
        self.m = nn.Sequential(
            *(Bottleneck(mid_channels, mid_channels, shortcut, expansion=1.0) for _ in range(repeats)))
        # After concatenation, 1x1 dimensional
        self.cv3 = Conv(2 * mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x2 = self.m(x2)
        # Feature stitching
        return self.cv3(torch.cat([x1, x2], dim=1))