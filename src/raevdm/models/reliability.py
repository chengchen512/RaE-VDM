from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ReliabilityEstimator(nn.Module):
    """Predict event-use confidence and low-event observability (not validity probabilities)."""

    def __init__(self, in_channels: int = 2, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(_groups(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(_groups(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.reliability_head = nn.Conv3d(hidden_channels, 2, 1)
        self.observability_head = nn.Conv3d(hidden_channels, 1, 1)

    def forward(self, event_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.net(event_raw)
        reliability = torch.sigmoid(self.reliability_head(feat))
        observability = torch.sigmoid(self.observability_head(feat))
        return {
            "features": feat,
            "reliability": reliability,
            "observability": observability,
        }


class ReliabilityCalibratedEventEncoder(nn.Module):
    """Seven-channel event condition in Eq. (18); legacy parameter names retained."""

    def __init__(
        self,
        feature_channels: int = 32,
        hidden_channels: int = 32,
        use_reliability: bool = True,
        use_observability: bool = True,
        cue_channels: int = 0,
        confidence_mode: str = "learned",
        confidence_threshold: float = 0.5,
        observability_mode: str = "learned",
    ) -> None:
        super().__init__()
        self.use_reliability = use_reliability
        self.use_observability = use_observability
        self.cue_channels = int(max(cue_channels, 0))
        if confidence_mode not in {"learned", "equal", "hard", "prior"}:
            raise ValueError("Unknown confidence_mode")
        if observability_mode not in {"learned", "prior", "naive", "zero"}:
            raise ValueError("Unknown observability_mode")
        self.confidence_mode = confidence_mode
        self.confidence_threshold = float(confidence_threshold)
        self.observability_mode = observability_mode
        self.reliability = ReliabilityEstimator(2 + self.cue_channels, hidden_channels)
        self.encoder = nn.Sequential(
            nn.Conv3d(7, feature_channels, 3, padding=1),
            nn.GroupNorm(_groups(feature_channels), feature_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels, 3, padding=1),
            nn.GroupNorm(_groups(feature_channels), feature_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels, 3, padding=1),
        )

    def forward(
        self,
        event_raw: torch.Tensor,
        target_frames: int,
        prior_reliability: torch.Tensor | None = None,
        prior_observability: torch.Tensor | None = None,
        auxiliary_cues: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self.cue_channels:
            if auxiliary_cues is None:
                raise ValueError("auxiliary_cues are required when cue_channels > 0.")
            if auxiliary_cues.shape[1] != self.cue_channels:
                raise ValueError(
                    f"Expected {self.cue_channels} cue channels, got {auxiliary_cues.shape[1]}."
                )
            estimator_input = torch.cat([event_raw, auxiliary_cues], dim=1)
        else:
            estimator_input = event_raw
        pred = self.reliability(estimator_input)
        reliability = pred["reliability"]
        observability = pred["observability"]

        if self.confidence_mode == "equal":
            reliability = torch.ones_like(reliability)
        elif self.confidence_mode == "hard":
            hard = (reliability >= self.confidence_threshold).to(reliability.dtype)
            # Hard decisions at inference; straight-through gradients when retraining.
            reliability = reliability + (hard - reliability).detach() if self.training else hard
        elif self.confidence_mode == "prior":
            if prior_reliability is None:
                raise ValueError("Prior-R requires event_prior_reliability")
            reliability = prior_reliability.to(reliability)
        if self.observability_mode == "prior":
            if prior_observability is None:
                raise ValueError("Prior-Obs requires event_prior_observability")
            observability = prior_observability.to(observability)
        elif self.observability_mode == "naive":
            observability = torch.ones_like(observability)
        elif self.observability_mode == "zero":
            observability = torch.zeros_like(observability)

        if not self.use_reliability:
            reliability = torch.ones_like(reliability)
        if not self.use_observability:
            observability = torch.zeros_like(observability)

        weighted = event_raw * reliability
        condition = torch.cat([event_raw, reliability, weighted, observability], dim=1)
        features = self.encoder(condition)
        if features.shape[2] != target_frames:
            features = F.interpolate(
                features,
                size=(target_frames, features.shape[-2], features.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )

        return {
            "features": features,
            "condition": condition,
            "reliability": reliability,
            "event_use_confidence": reliability,
            "observability": observability,
            "low_event_observability": observability,
            "prior_reliability": prior_reliability,
            "prior_observability": prior_observability,
            "auxiliary_cues": auxiliary_cues,
        }


def collapse_event_steps(
    tensor: torch.Tensor,
    clip_len: int,
    event_bins: int,
    reduce: str = "mean",
) -> torch.Tensor:
    """Collapse [(T-1)*B] event steps to [T-1] frame intervals."""
    b, c, steps, h, w = tensor.shape
    expected = (clip_len - 1) * event_bins
    if steps != expected:
        # Fallback: interpolate temporal evidence to T-1 intervals.
        out = F.interpolate(tensor, size=(clip_len - 1, h, w), mode="trilinear", align_corners=False)
        return out
    out = tensor.view(b, c, clip_len - 1, event_bins, h, w)
    if reduce == "sum":
        return out.sum(dim=3)
    if reduce == "max":
        return out.max(dim=3).values
    return out.mean(dim=3)
