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
        # (B, C, H // 2, W // 2)
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
        # K/V: 单头共享投影（与所有Q头共享）
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

