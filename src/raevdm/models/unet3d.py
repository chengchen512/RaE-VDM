from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1))
        emb = t.float()[:, None] * freq[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb))[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DownsampleSpatial(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalVideoUNet(nn.Module):
    """Lightweight conditional 3D U-Net for video diffusion."""

    def __init__(
        self,
        in_channels: int = 1,
        condition_channels: int = 32,
        extra_condition_channels: int = 0,
        base_channels: int = 32,
        channel_mult: Iterable[int] = (1, 2, 4),
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.extra_condition_channels = int(max(extra_condition_channels, 0))
        input_channels = in_channels + 1 + 1 + condition_channels + self.extra_condition_channels
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input = nn.Conv3d(input_channels, base_channels, 3, padding=1)

        mults = list(channel_mult)
        channels = [base_channels * m for m in mults]
        self.downs = nn.ModuleList()
        prev = base_channels
        for i, ch in enumerate(channels):
            self.downs.append(
                nn.ModuleDict(
                    {
                        "block1": ResidualBlock3D(prev, ch, time_embed_dim),
                        "block2": ResidualBlock3D(ch, ch, time_embed_dim),
                        "down": DownsampleSpatial(ch) if i < len(channels) - 1 else nn.Identity(),
                    }
                )
            )
            prev = ch

        self.mid1 = ResidualBlock3D(prev, prev, time_embed_dim)
        self.mid2 = ResidualBlock3D(prev, prev, time_embed_dim)

        self.ups = nn.ModuleList()
        for ch in reversed(channels):
            self.ups.append(
                nn.ModuleDict(
                    {
                        "block1": ResidualBlock3D(prev + ch, ch, time_embed_dim),
                        "block2": ResidualBlock3D(ch, ch, time_embed_dim),
                    }
                )
            )
            prev = ch

        self.out_norm = nn.GroupNorm(_groups(prev), prev)
        self.out = nn.Conv3d(prev, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_mask: torch.Tensor,
        extra_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temb = self.time_mlp(t)
        inputs = [x, keyframes, keyframe_mask, condition]
        if self.extra_condition_channels > 0:
            if extra_condition is None:
                b, _, frames_t, h_, w_ = x.shape
                extra_condition = x.new_zeros((b, self.extra_condition_channels, frames_t, h_, w_))
            inputs.append(extra_condition)
        h = torch.cat(inputs, dim=1)
        h = self.input(h)

        skips: List[torch.Tensor] = []
        for stage in self.downs:
            h = stage["block1"](h, temb)
            h = stage["block2"](h, temb)
            skips.append(h)
            h = stage["down"](h)

        h = self.mid1(h, temb)
        h = self.mid2(h, temb)

        for stage in self.ups:
            skip = skips.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = stage["block1"](h, temb)
            h = stage["block2"](h, temb)

        return self.out(F.silu(self.out_norm(h)))
