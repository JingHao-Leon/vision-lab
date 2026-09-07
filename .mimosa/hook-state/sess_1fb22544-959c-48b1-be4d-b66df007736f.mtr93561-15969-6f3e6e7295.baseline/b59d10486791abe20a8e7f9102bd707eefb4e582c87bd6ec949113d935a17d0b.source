"""Vision Transformer from scratch: patch embed → [class] token → pre-norm
transformer blocks → classification head.

Design choices that matter on CIFAR-10-scale data:
- 4×4 patches on 32×32 images → 64 tokens (8×8 grid), no downsampled backbone
- learned positional embeddings (simpler than 2D RoPE, equivalent at this size)
- pre-norm blocks (GPT style) — trains noticeably steadier than post-norm
- stochastic depth + heavy augmentation live in the training script, not here
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch: int = 4, in_ch: int = 3, dim: int = 256):
        super().__init__()
        assert img_size % patch == 0
        self.grid = img_size // patch
        self.n_patches = self.grid * self.grid
        # a conv with kernel=stride=patch IS the patch embedding (Vit paper 0.2)
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, n_patches, dim)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.h, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.h, self.hd).transpose(1, 2)
        k = k.view(B, N, self.h, self.hd).transpose(1, 2)
        v = v.view(B, N, self.h, self.hd).transpose(1, 2)
        # F.scaled_dot_product_attention: fused memory-efficient kernel
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(B, N, -1)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.n2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class ViT(nn.Module):
    def __init__(self, img_size: int = 32, patch: int = 4, in_ch: int = 3,
                 n_classes: int = 10, dim: int = 256, depth: int = 6,
                 heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1,
                 emb_dropout: float = 0.1):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch, in_ch, dim)
        n = self.patch.n_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.drop = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(x)  # (B, N, dim)
        x = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        x = self.drop(x + self.pos)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x)[:, 0])  # classify from the [class] token

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
