from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class FusionResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class FlowResidualFusionHead(nn.Module):
    """Occlusion-aware fusion of warped and residual-corrected videos.

    The final layer is zero initialized, so the initial prediction exactly
    follows the configured time-adaptive RAFT/residual blend.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        refinement_scale: float = 0.05,
        alpha_delta_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.refinement_scale = float(refinement_scale)
        self.alpha_delta_scale = float(alpha_delta_scale)
        # residual, warp, linear, three pairwise differences, event gate,
        # warp confidence, and normalized middle-frame time.
        in_channels = 9
        self.input = nn.Conv3d(in_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            FusionResidualBlock3D(hidden_channels),
            FusionResidualBlock3D(hidden_channels),
            FusionResidualBlock3D(hidden_channels),
        )
        self.out_norm = nn.GroupNorm(_groups(hidden_channels), hidden_channels)
        self.out = nn.Conv3d(hidden_channels, 2, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        residual_video: torch.Tensor,
        warped_video: torch.Tensor,
        linear_video: torch.Tensor,
        event_gate: torch.Tensor,
        warp_confidence: torch.Tensor,
        time_map: torch.Tensor,
        keyframe_mask: torch.Tensor,
        base_raft_peak: float = 0.55,
    ) -> dict[str, torch.Tensor]:
        features = torch.cat(
            [
                residual_video,
                warped_video,
                linear_video,
                (residual_video - warped_video).abs(),
                (residual_video - linear_video).abs(),
                (warped_video - linear_video).abs(),
                event_gate,
                warp_confidence,
                time_map,
            ],
            dim=1,
        )
        h = self.input(features)
        h = self.blocks(h)
        raw = self.out(F.silu(self.out_norm(h)))

        alpha_delta = torch.tanh(raw[:, :1]) * self.alpha_delta_scale
        refinement = torch.tanh(raw[:, 1:2]) * self.refinement_scale

        time_strength = 4.0 * time_map * (1.0 - time_map)
        base_alpha = (float(base_raft_peak) * time_strength).clamp(0.02, 0.98)
        base_logit = torch.log(base_alpha) - torch.log1p(-base_alpha)
        alpha = torch.sigmoid(base_logit + alpha_delta)
        alpha = alpha * (1.0 - keyframe_mask)

        fused_base = alpha * warped_video + (1.0 - alpha) * residual_video
        pred_raw = fused_base + refinement
        pred = pred_raw.clamp(0.0, 1.0)
        pred = pred * (1.0 - keyframe_mask) + residual_video * keyframe_mask
        return {
            "pred": pred,
            "pred_raw": pred_raw,
            "fused_base": fused_base,
            "alpha": alpha,
            "alpha_delta": alpha_delta,
            "refinement": refinement,
        }
