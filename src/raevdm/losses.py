from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .models.raevdm import diffusion_to_intensity, video_to_bcthw
from .models.reliability import collapse_event_steps


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def _event_bins_from_batch(batch: Dict[str, Any], default: int) -> int:
    value = batch.get("event_bins", default)
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    return int(value)


def compute_losses(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    event_bins: int,
    charbonnier_eps: float = 1e-3,
    theory_cfg: Dict[str, Any] | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    theory_cfg = theory_cfg or {}
    theory_aligned = bool(theory_cfg.get("enabled", False))
    eps_loss = F.mse_loss(outputs["eps_pred"], outputs["noise"])

    pred = diffusion_to_intensity(outputs["x0_pred"])
    pred_raw = diffusion_to_intensity(outputs.get("x0_pred_raw", outputs["x0_pred"]))
    gt = video_to_bcthw(batch["frames"].float())
    key_mask = video_to_bcthw(batch["keyframe_mask"].float())

    log_pred = torch.log(pred.clamp_min(1e-4))
    pred_delta = log_pred[:, :, 1:] - log_pred[:, :, :-1]
    target_delta = video_to_bcthw(batch["event_delta"].float())

    clip_len = gt.shape[2]
    event_bins = _event_bins_from_batch(batch, event_bins)
    raw = batch["event_raw"].float()
    learned_rel = outputs["event"]["reliability"]
    rel_score_steps = (learned_rel * raw).sum(dim=1, keepdim=True) / raw.sum(dim=1, keepdim=True).clamp_min(1e-6)
    rel_interval = collapse_event_steps(rel_score_steps, clip_len=clip_len, event_bins=event_bins, reduce="mean")
    prior_precision = video_to_bcthw(batch["event_interval_precision"].float())
    if theory_aligned:
        precision = rel_interval
        counts = batch.get("event_counts")
        if counts is not None and bool(theory_cfg.get("weighted_event_target", True)):
            b, _, steps, h, w = counts.shape
            expected = (clip_len - 1) * event_bins
            if steps != expected:
                raise ValueError(f"event_counts has {steps} steps, expected {expected}.")
            weighted_counts = counts.float() * learned_rel
            weighted_counts = weighted_counts.view(b, 2, clip_len - 1, event_bins, h, w)
            signed_counts = weighted_counts[:, 0].sum(dim=2) - weighted_counts[:, 1].sum(dim=2)
            contrast = batch.get("contrast_threshold", 0.2)
            if torch.is_tensor(contrast):
                contrast = contrast.to(signed_counts).reshape(b, 1, 1, 1)
            else:
                contrast = float(contrast)
            target_delta = (contrast * signed_counts.unsqueeze(1)).clamp(-2.0, 2.0)
        if bool(theory_cfg.get("detach_precision", False)):
            precision = precision.detach()
    else:
        precision = (0.5 * rel_interval + 0.5 * prior_precision).detach()

    physics_residual = pred_delta - target_delta
    physics_loss = (precision * charbonnier(physics_residual, charbonnier_eps)).sum() / precision.sum().clamp_min(1.0)

    learned_obs = collapse_event_steps(outputs["event"]["observability"], clip_len=clip_len, event_bins=event_bins, reduce="mean")
    prior_obs = video_to_bcthw(batch["event_interval_observability"].float())
    if theory_aligned:
        observability = learned_obs
        quiet_weight = observability * (1.0 - precision).clamp(0.0, 1.0)
        quiet_margin = float(theory_cfg.get("quiet_margin", 0.02))
        quiet_penalty = charbonnier(
            F.relu(torch.abs(pred_delta) - quiet_margin), charbonnier_eps
        )
        quiet_loss = (quiet_weight * quiet_penalty).sum() / quiet_weight.sum().clamp_min(1.0)
    else:
        observability = (0.5 * learned_obs + 0.5 * prior_obs).detach()
        quiet_loss = (observability * charbonnier(pred_delta, charbonnier_eps)).sum() / observability.sum().clamp_min(1.0)

    anchor_loss = (key_mask * torch.abs(pred_raw - gt)).sum() / key_mask.sum().clamp_min(1.0)
    unknown_mask = 1.0 - key_mask
    reconstruction_loss = (
        unknown_mask * charbonnier(pred - gt, charbonnier_eps)
    ).sum() / unknown_mask.sum().clamp_min(1.0)

    if pred.shape[2] >= 3:
        second = pred[:, :, 2:] - 2.0 * pred[:, :, 1:-1] + pred[:, :, :-2]
        temporal_loss = charbonnier(second, charbonnier_eps).mean()
    else:
        temporal_loss = pred.new_tensor(0.0)

    correctness = torch.exp(-torch.abs(physics_residual).detach()).clamp(0.0, 1.0)
    calibration_loss = F.mse_loss(rel_interval, correctness)

    prior_rel = batch.get("event_prior_reliability")
    if prior_rel is not None:
        prior_loss = F.mse_loss(learned_rel, prior_rel.float())
    else:
        prior_loss = pred.new_tensor(0.0)

    losses = {
        "diffusion": eps_loss,
        "reconstruction": reconstruction_loss,
        "event_physics": physics_loss,
        "quiet": quiet_loss,
        "anchor": anchor_loss,
        "temporal": temporal_loss,
        "reliability_calibration": calibration_loss,
        "reliability_prior": prior_loss,
    }
    total = pred.new_tensor(0.0)
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    losses["total"] = total
    return total, losses
