import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialReduction(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=2):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=dim
        )

    def forward(self, x):
        # (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # downsampling: (B, C, H // 2, W // 2)
        x = self.conv(x)
        # (B, H//2, W//2, C)
        x = x.permute(0, 2, 3, 1)
        return x


class MobileMQA(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, spatial_reduction=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # K/V: Single-head shared projection (shared with all Q heads)
        self.k_proj = nn.Linear(dim, self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.head_dim, bias=qkv_bias)

        self.spatial_reduction = spatial_reduction
        if spatial_reduction:
            self.k_reduction = SpatialReduction(dim=self.head_dim)
            self.v_reduction = SpatialReduction(dim=self.head_dim)

        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # (B, H, W, C)
        batch_size, H, W, _ = x.shape
        seq_len = H * W

        # (B, H, W, dim)
        q = self.q_proj(x)
        # (B, H, W, head_dim)
        k = self.k_proj(x)
        # (B, H, W, head_dim)
        v = self.v_proj(x)

        # Reduction of application space for K/V
        if self.spatial_reduction:
            # (B, H//2, W//2, head_dim)
            k = self.k_reduction(k)
            # (B, H//2, W//2, head_dim)
            v = self.v_reduction(v)
            # Flattening the spatial dimensions after downsampling
            kv_seq_len = (H // 2) * (W // 2)
            # (B, kv_seq_len, head_dim)
            k = k.flatten(1, 2)
            # (B, kv_seq_len, head_dim)
            v = v.flatten(1, 2)
        else:
            # Don't reduce, simply flatten the original spatial dimensions directly.
            kv_seq_len = seq_len
            # (B, seq_len, head_dim)
            k = k.flatten(1, 2)
            v = v.flatten(1, 2)

        # (B, seq_len, dim)
        q = q.flatten(1, 2)
        # (B, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # (B, num_heads, kv_seq_len, head_dim)
        k = k.view(batch_size, kv_seq_len, 1, self.head_dim).permute(0, 2, 1, 3).expand(-1, self.num_heads, -1, -1)
        # (B, num_heads, kv_seq_len, head_dim)
        v = v.view(batch_size, kv_seq_len, 1, self.head_dim).permute(0, 2, 1, 3).expand(-1, self.num_heads, -1, -1)

        # (B, num_heads, seq_len, kv_seq_len)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        # (B, num_heads, seq_len, head_dim)
        out = attn @ v

        # (B, seq_len, dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.dim)
        # (B, seq_len, dim)
        out = self.out_proj(out)
        out = out.view(batch_size, H, W, self.dim)

        return out



class GAM(nn.Module):
    def __init__(self, channels, rate=4):
        super(GAM, self).__init__()
        mid_channels = channels // rate

        self.channel_attention = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # channel attention
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att
        # spatial attention
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 4):
        super(ResBlock_CBAM,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )
        self.cbam = CBAM(channel=places*self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out



class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))

        self.excitation = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Read the number of batch data images and the number of channels
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1  # Choose whether to subtract the central pixel, but usually it is not subtracted.

        # Centralized feature map
        mu = x.mean(dim=[2, 3], keepdim=True).expand_as(x)
        x_centered = x - mu

        x_minus_mu_square = x_centered.pow(2)

        # Normalize and calculate the attention weights
        norm_factor = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda
        y = x_minus_mu_square / (4 * norm_factor) + 0.5
        attention_map = self.activaton(y)

        return x * attention_map