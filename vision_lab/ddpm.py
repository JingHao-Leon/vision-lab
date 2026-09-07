"""DDPM (Ho et al. 2020) from scratch: linear beta schedule, ε-prediction UNet,
ancestral sampling.

Everything is dimension-agnostic so the same code trains on 28×28 MNIST and
32×32 CIFAR; only the channel count changes. Kept deliberately small (2.7M
params) — trains to recognizable digits on a 3090 in ~15 minutes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """The transformer position-encoding formula, reused for diffusion steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.emb = SinusoidalTimeEmbedding(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(self.emb(t))


class ResidualBlock(nn.Module):
    """Conv block with a time embedding injected FiLM-style (scale+shift)."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(),
                                   nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.conv2 = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(),
                                   nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x) + self.t_proj(t_emb)[:, :, None, None]
        return self.skip(x) + self.conv2(h)


class AttentionBlock(nn.Module):
    """Single-head self-attention over spatial positions (at low resolution only)."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).permute(1, 0, 3, 2).unbind(0)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return x + self.proj(y.reshape(B, C, H, W))


class UNet(nn.Module):
    """Down 32→16→8, mid attention at 8×8, up-symmetry with skip connections."""

    def __init__(self, in_ch: int = 1, base: int = 64, t_dim: int = 256):
        super().__init__()
        self.t_mlp = TimeMLP(t_dim)
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        self.d1 = ResidualBlock(base, base, t_dim)
        self.down1 = nn.Conv2d(base, base, 3, stride=2, padding=1)  # 32→16 (or 28→14)
        self.d2 = ResidualBlock(base, base * 2, t_dim)
        self.down2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)  # →8 (→7)
        self.mid1 = ResidualBlock(base * 2, base * 2, t_dim)
        self.mid_attn = AttentionBlock(base * 2)
        self.mid2 = ResidualBlock(base * 2, base * 2, t_dim)

        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, stride=2, padding=1)
        self.u2 = ResidualBlock(base * 4, base, t_dim)
        self.up1 = nn.ConvTranspose2d(base, base, 4, stride=2, padding=1)
        self.u1 = ResidualBlock(base * 2, base, t_dim)
        self.attn1 = AttentionBlock(base)
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(),
                                 nn.Conv2d(base, in_ch, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(t)
        s = self.stem(x)
        h1 = self.d1(s, t_emb)                       # base, full res
        h2 = self.d2(self.down1(h1), t_emb)          # 2x, half
        m = self.mid2(self.mid_attn(self.mid1(self.down2(h2), t_emb)), t_emb)  # 4x
        u2 = self.u2(torch.cat([h2, self.up2(m)], dim=1), t_emb)
        u1 = self.attn1(self.u1(torch.cat([h1, self.up1(u2)], dim=1), t_emb))
        return self.out(u1)


class DDPM:
    """Linear-β DDPM: q(x_t|x_0) closed form + ε-prediction training objective."""

    def __init__(self, timesteps: int = 400, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: torch.device | str = "cpu"):
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1mac = torch.sqrt(1.0 - self.alphas_cumprod)
        self.betas, self.alphas = betas, alphas

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: x_t = √ᾱ·x0 + √(1-ᾱ)·ε (closed form)."""
        if noise is None:
            noise = torch.randn_like(x0)
        xt = (self.sqrt_ac[t][:, None, None, None] * x0
              + self.sqrt_1mac[t][:, None, None, None] * noise)
        return xt, noise

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, img_size: int,
               in_ch: int = 1) -> torch.Tensor:
        """Ancestral sampling: start from pure noise, denoise T→0 with the
        posterior mean + per-step noise (σ² = β)."""
        model.eval()
        x = torch.randn(n, in_ch, img_size, img_size, device=next(model.parameters()).device)
        for t in reversed(range(self.T)):
            tb = torch.full((n,), t, device=x.device, dtype=torch.long)
            pred_noise = model(x, tb)
            alpha, beta = self.alphas[t], self.betas[t]
            ac = self.alphas_cumprod[t]
            mean = (1 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - ac) * pred_noise)
            if t > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        model.train()
        return torch.clamp(x, -1, 1)
