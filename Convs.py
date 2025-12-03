import torch.nn as nn
import torch
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Conv2d
    BatchNorm2d
    SiLU activation
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding, dilation),
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # YOLOv11默认激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.dconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            groups=in_channels,  # When groups=in_channel, it indicates performing per-channel convolution.
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


class ChannelReconstruction(nn.Module):
    '''
    Channel reconstruction branch
    '''

    def __init__(self, in_channels, groups=4, reduction_ratio=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        # Number of channels per group
        self.ch_per_group = in_channels // groups
        # Number of channels after dimensionality reduction
        self.reduced_ch = int(self.ch_per_group * reduction_ratio)

        self.reduce_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=groups * self.reduced_ch,
            kernel_size=1,
            groups=groups,
            bias=False
        )

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling (B, G*rC, 1, 1)
            nn.Conv2d(
                in_channels=groups * self.reduced_ch,
                out_channels=groups * self.reduced_ch,  # Maintain Channel
                kernel_size=1,
                groups=groups,  # Group attention to prevent cross-group interference
                bias=False
            ),
            nn.Sigmoid()  # Generate channel weights
        )

        self.expand_conv = nn.Conv2d(
            in_channels=groups * self.reduced_ch,
            out_channels=in_channels,
            kernel_size=1,
            groups=groups,  # Group-wise dimensionality expansion while preserving channel independence
            bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, H, W)
        residual = x  # Residual connection

        # Group-wise dimension reduction
        x = self.reduce_conv(x)  # (B, G*rC, H, W)
        x = self.relu(x)

        # Channel attention weighting
        attn = self.attn(x)  # (B, G*rC, 1, 1)
        x = x * attn  # (B, G*rC, H, W)

        # Group-wise dimension-raising, restoring the number of channels
        x = self.expand_conv(x)  # (B, C, H, W)

        # Residual connection, preserve original features
        return x + residual


class SpatialReconstruction(nn.Module):
    """
    Spatial reconstruction branch
    """

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels

        self.downsample = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )

        self.attn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,  # Single-channel spatial weight map
                kernel_size=1,  # 1x1 convolutional compression channel
                bias=False
            ),
            nn.Sigmoid()  # Generate spatial weights
        )
        self.upsample_mode = "bilinear"

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        residual = x  # 残差连接

        # (B, C, H, W) → (B, C, H//2, W//2)
        x = self.downsample(x)

        # (B, C, H//2, W//2) → (B, 1, H//2, W//2)
        attn = self.attn(x)  # (B, 1, H//2, W//2)
        x = x * attn  # (B, C, H//2, W//2)

        # (B, C, H//2, W//2) → (B, C, H, W)
        x = F.interpolate(
            x,
            size=(H, W),
            mode=self.upsample_mode,
            align_corners=False
        )

        # Residual connection, preserving original spatial features
        return x + residual


class SCConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4, reduction_ratio=0.5, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Channel reconstruction branch
        self.channel_recon = ChannelReconstruction(
            in_channels=in_channels,
            groups=groups,
            reduction_ratio=reduction_ratio
        )
        # Spatial reconstruction branch
        self.spatial_recon = SpatialReconstruction(
            in_channels=in_channels,
            kernel_size=kernel_size
        )
        # Fusion convolution, integrates dual-channel information
        # (optional; in lightweight scenarios, direct addition is sufficient)
        self.fusion = nn.Conv2d(
            in_channels=in_channels * 2,  # Concatenate the outputs of two branches
            out_channels=in_channels,  # Restore original channel count
            kernel_size=1,
            bias=False
        )

        self.channel_trans = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

        if out_channels != in_channels:
            self.residual_adjust = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            )
        else:
            self.residual_adjust = nn.Identity()

    def forward(self, x):
        residual = x

        channel_feat = self.channel_recon(x)  # (B, C, H, W)
        spatial_feat = self.spatial_recon(x)  # (B, C, H, W)

        fused = torch.cat([channel_feat, spatial_feat], dim=1)  # (B, 2C, H, W)
        fused = self.fusion(fused)  # (B, C, H, W)

        out = self.channel_trans(fused)

        residual = self.residual_adjust(residual)

        return out + residual
