# utils/vit.py

import torch
import torch.nn as nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, dropout=0.):
        super().__init__()
        self.heads = heads
        dim_head = dim // heads if dim_head is None else dim_head
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # reshape for multi‑head
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(dots, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """
    Vision Transformer

    Args:
      image_size      : 输入的图像尺寸（假设正方形），实际接收的是 patch_dim=1 时把特征图当 “图像”。
      in_channels     : 输入通道数（你的代码中是 out_channels*8）。
      dim             : Transformer 的 hidden dim（等于 in_channels）。
      depth           : Transformer block 数量。
      heads           : 注意力头数。
      mlp_dim         : Transformer MLP 隐藏层宽度。
      patch_dim       : 切 patch 的大小（你的场景下一般设为 1）。
      dropout         : Dropout 比例（可留默认）。
      classification  : 是否在最后接 classification head；我们设为 False，只输出特征序列。
    """
    def __init__(self,
                 image_size,
                 in_channels,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 patch_dim=16,
                 dropout=0.,
                 classification=True):
        super().__init__()
        assert image_size % patch_dim == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_dim) ** 2
        self.patch_dim = patch_dim

        # 从特征图到 patch embedding
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_dim, stride=patch_dim),
            rearrange('b c h w -> b (h w) c')
        )

        # 可学习的位置编码 + class token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.norm = nn.LayerNorm(dim)
        self.classification = classification
        if self.classification:
            self.mlp_head = nn.Linear(dim, dim)

    def forward(self, img):
        # img: (b, in_channels, H, W)
        x = self.to_patch_embedding(img)               # (b, num_patches, dim)
        b, n, _ = x.shape

        # 拼接 class token
        cls_tokens = self.class_token.expand(b, -1, -1) # (b,1,dim)
        x = torch.cat((cls_tokens, x), dim=1)           # (b, num_patches+1, dim)

        x = x + self.pos_embedding[:, :n+1]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)

        if self.classification:
            # 返回分类向量
            return self.mlp_head(x[:, 0])
        # 返回所有 patch（不含 class token），用于 reshape 回特征图
        return x[:, 1:]
