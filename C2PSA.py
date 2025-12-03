import torch.nn as nn
import torch
import torch.nn.functional as F
from Convs import Conv
from Bottleneck import Bottleneck


class PSA(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Attention weight generation: Generate weights for the three branches
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, 3)
        )
        self.softmax = nn.Softmax(dim=1)

        # Define three convolution operations with different receptive fields
        self.conv1 = nn.Conv2d(channels // 3, channels // 3, kernel_size=3, padding=1, groups=channels // 3)
        self.conv2 = nn.Conv2d(channels // 3, channels // 3, kernel_size=5, padding=2, groups=channels // 3)
        self.conv3 = nn.Conv2d(channels - 2 * (channels // 3), channels - 2 * (channels // 3), kernel_size=7, padding=3,
                               groups=channels - 2 * (channels // 3))


def forward(self, x):
    B, C, H, W = x.shape

    # Calculate global features
    avg_feat = self.avg_pool(x).view(B, C)

    # Generate three attention weights
    weights = self.fc(avg_feat)
    weights = self.softmax(weights).unsqueeze(-1).unsqueeze(-1)

    # Divide the input feature map into three parts along the channel dimension
    c1 = C // 3
    c2 = C // 3
    c3 = C - c1 - c2

    x1, x2, x3 = torch.split(x, [c1, c2, c3], dim=1)

    # Apply different convolution and attention weights to each part
    x1 = self.conv1(x1) * weights[:, 0:1, :, :]
    x2 = self.conv2(x2) * weights[:, 1:2, :, :]
    x3 = self.conv3(x3) * weights[:, 2:3, :, :]

    # The result of the concatenation
    return torch.cat([x1, x2, x3], dim=1)


class C2PSA(nn.Module):
    def __init__(self, in_channels, out_channels, repeats=1, shortcut=True, expansion=0.5):
        super().__init__()
        # Number of intermediate layer channels
        mid_channels = int(out_channels * expansion)
        # 1x1 dimensionality reduction
        self.cv1 = Conv(in_channels, mid_channels, kernel_size=1)
        # 1x1 dimensionality reduction
        self.cv2 = Conv(in_channels, mid_channels, kernel_size=1)
        # Stacked Bottleneck + PSA Attention
        self.m = nn.Sequential(
            *(Bottleneck(mid_channels, mid_channels, shortcut, expansion=1.0) for _ in range(repeats)),
            PSA(mid_channels)  # Add PSA attention
        )
        self.cv3 = Conv(2 * mid_channels, out_channels, kernel_size=1)  # After concatenation, 1x1 dimensional

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x2 = self.m(x2)
        return self.cv3(torch.cat([x1, x2], dim=1))
