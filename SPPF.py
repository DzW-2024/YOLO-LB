import torch
import torch.nn as nn
from Convs import Conv, DWConv, GhostConv, SCConv

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        mid_channels = in_channels // 2  # Number of intermediate layer channels (halved)
        self.cv1 = Conv(in_channels, mid_channels, kernel_size=1)  # 1x1 dimensionality reduction
        self.cv2 = Conv(mid_channels * 4, out_channels, kernel_size=1)  # After concatenation, 1x1 dimensional
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)  # 5x5 pooling

    def forward(self, x):
        x = self.cv1(x)
        # Four times of pooling feature concatenation (simulating multi-scale receptive fields)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))