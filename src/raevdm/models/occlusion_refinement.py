from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SpatialTemporalBlock(nn.Module):
    """Factorized 3D residual block with inexpensive temporal context."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.spatial = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, dilation, dilation),
            dilation=(1, dilation, dilation),
        )
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.temporal = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.spatial(F.silu(self.norm1(x)))
        h = self.temporal(F.silu(self.norm2(h)))
        return x + h


class OcclusionAwareRefinementHead(nn.Module):
    """Multi-scale repair head for flow occlusions and motion boundaries.

    The output projection is zero initialized. A newly attached head therefore
    reproduces the incoming flow-fusion prediction exactly before training.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        refinement_scale: float = 0.06,
        branch_delta_scale: float = 0.30,
    ) -> None:
        super().__init__()
        self.refinement_scale = float(refinement_scale)
        self.branch_delta_scale = float(branch_delta_scale)

        # base, residual, aggregate warp, linear, two endpoint warps,
        # base/residual disagreement, endpoint disagreement, event gate,
        # confidence, two visibility maps, occlusion, hole, motion boundary,
        # and normalized interpolation time.
        in_channels = 16
        low_channels = hidden_channels * 2
        self.stem = nn.Conv3d(in_channels, hidden_channels, 3, padding=1)
        self.full_blocks = nn.Sequential(
            SpatialTemporalBlock(hidden_channels),
            SpatialTemporalBlock(hidden_channels, dilation=2),
        )
        self.down = nn.Conv3d(
            hidden_channels,
            low_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.low_blocks = nn.Sequential(
            SpatialTemporalBlock(low_channels),
            SpatialTemporalBlock(low_channels, dilation=2),
            SpatialTemporalBlock(low_channels),
        )
        self.merge = nn.Conv3d(hidden_channels + low_channels, hidden_channels, 3, padding=1)
        self.merge_blocks = nn.Sequential(
            SpatialTemporalBlock(hidden_channels),
            SpatialTemporalBlock(hidden_channels),
        )
        self.out_norm = nn.GroupNorm(_groups(hidden_channels), hidden_channels)
        # Four branch deltas (residual, endpoint-0, endpoint-1, linear) and
        # one direct local refinement.
        self.out = nn.Conv3d(hidden_channels, 5, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        base_prediction: torch.Tensor,
        residual_video: torch.Tensor,
        warped_video: torch.Tensor,
        linear_video: torch.Tensor,
        endpoint_warp0: torch.Tensor,
        endpoint_warp1: torch.Tensor,
        event_gate: torch.Tensor,
        warp_confidence: torch.Tensor,
        visibility0: torch.Tensor,
        visibility1: torch.Tensor,
        occlusion_map: torch.Tensor,
        hole_map: torch.Tensor,
        motion_boundary: torch.Tensor,
        time_map: torch.Tensor,
        keyframe_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        endpoint_disagreement = (endpoint_warp0 - endpoint_warp1).abs()
        features = torch.cat(
            [
                base_prediction,
                residual_video,
                warped_video,
                linear_video,
                endpoint_warp0,
                endpoint_warp1,
                (base_prediction - residual_video).abs(),
                endpoint_disagreement,
                event_gate,
                warp_confidence,
                visibility0,
                visibility1,
                occlusion_map,
                hole_map,
                motion_boundary,
                time_map,
            ],
            dim=1,
        )

        skip = self.full_blocks(self.stem(features))
        low = self.low_blocks(self.down(skip))
        low = F.interpolate(
            low,
            size=skip.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        h = self.merge_blocks(self.merge(torch.cat([skip, low], dim=1)))
        raw = self.out(F.silu(self.out_norm(h)))

        uncertainty = torch.maximum(1.0 - warp_confidence, occlusion_map)
        uncertainty = torch.maximum(uncertainty, hole_map)
        focus = torch.maximum(uncertainty, motion_boundary)
        focus = torch.maximum(focus, endpoint_disagreement.clamp(0.0, 1.0))
        focus = focus.clamp(0.0, 1.0) * (1.0 - keyframe_mask)

        branch_deltas = (
            torch.tanh(raw[:, :4])
            * self.branch_delta_scale
            * focus
        )
        branches = torch.stack(
            [residual_video, endpoint_warp0, endpoint_warp1, linear_video],
            dim=1,
        )
        base_expanded = base_prediction[:, None]
        branch_adjustment = (
            branch_deltas[:, :, None] * (branches - base_expanded)
        ).sum(dim=1)

        refinement_focus = (0.20 + 0.80 * torch.maximum(focus, event_gate)).clamp(0.0, 1.0)
        refinement = (
            torch.tanh(raw[:, 4:5])
            * self.refinement_scale
            * refinement_focus
        )
        pred_raw = base_prediction + branch_adjustment + refinement
        pred = pred_raw.clamp(0.0, 1.0)
        pred = pred * (1.0 - keyframe_mask) + base_prediction * keyframe_mask
        return {
            "pred": pred,
            "pred_raw": pred_raw,
            "branch_deltas": branch_deltas,
            "branch_adjustment": branch_adjustment,
            "refinement_v2": refinement,
            "refinement_focus": focus,
        }
