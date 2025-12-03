import torch.nn as nn
from Convs import Conv, DWConv, GhostConv, SCConv

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        mid_channels = int(out_channels * expansion)  # Number of intermediate-level channels (bottleneck ratio)
        self.cv1 = Conv(in_channels, mid_channels, kernel_size=1)  # 1x1 dimensionality reduction
        self.cv2 = Conv(mid_channels, out_channels, kernel_size=3)  # 3x3 convolution
        self.shortcut = shortcut and in_channels == out_channels  # Residual connection condition

    def forward(self, x):
        residual = x
        x = self.cv1(x)
        x = self.cv2(x)
        if self.shortcut:
            x += residual
        return x