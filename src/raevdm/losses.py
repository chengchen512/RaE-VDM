"""Stage objectives corresponding to Sec. III-H, Eqs. (31)--(40)."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .event_statistics import count_priors, learned_event_statistics, signed_event_delta
from .models.raevdm import diffusion_to_intensity, video_to_bcthw


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x.square() + eps * eps)


def log_intensity_delta(video: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    log_video = torch.log(video.float() + eps)
    return log_video[:, :, 1:] - log_video[:, :, :-1]


def compute_losses(batch: Dict[str, torch.Tensor], outputs: Dict[str, Any],
                   weights: Dict[str, float], event_bins: int,
                   charbonnier_eps: float = 1e-3,
                   theory_cfg: Dict[str, Any] | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    theory_cfg = theory_cfg or {}
    reference = batch["frames"]
    zero = reference.new_zeros(())
    stochastic = outputs.get("diffusion_outputs", outputs)
    diffusion = F.mse_loss(stochastic["eps_pred"].float(), stochastic["noise"].float()) \
        if "eps_pred" in stochastic else zero
    names = ("reconstruction", "event_physics", "quiet", "anchor", "temporal",
             "reliability_calibration", "reliability_prior", "reliability_supervision")
    losses = {name: zero for name in names}
    losses["diffusion"] = diffusion
    if not any(float(weights.get(name, 0.0)) for name in names):
        total = float(weights.get("diffusion", 0.0)) * diffusion
        losses["total"] = total
        return total, losses

    if "pred" in outputs:
        pred = outputs["pred"].float()
        raw_pred = outputs.get("pred_raw", pred).float()
    elif "residual" in outputs:
        pred = outputs["residual"]["pred"].float()
        raw_pred = outputs["residual"]["pred_raw"].float()
    else:
        pred = diffusion_to_intensity(outputs["x0_pred"].float())
        raw_pred = (outputs.get("x0_pred_raw", outputs["x0_pred"]).float() + 1.0) / 2.0
    gt = video_to_bcthw(reference.float())
    losses["reconstruction"] = F.l1_loss(pred, gt)
    losses["anchor"] = (raw_pred[:, :, 0] - gt[:, :, 0]).abs().mean() + \
        (raw_pred[:, :, -1] - gt[:, :, -1]).abs().mean()
    losses["temporal"] = (pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]).abs().mean() \
        if pred.shape[2] > 2 else zero

    counts = batch.get("event_counts")
    if counts is None:
        raise ValueError("Paper losses require raw event_counts, not normalized event_raw.")
    bins_value = batch.get("event_bins", event_bins)
    bins = int(bins_value.flatten()[0].item()) if torch.is_tensor(bins_value) else int(bins_value)
    if counts.shape[2] != (gt.shape[2] - 1) * bins:
        raise ValueError("Event count intervals do not match reconstruction timestamps.")
    event = outputs["event"]
    rel = event["reliability"].float()
    contrast = batch.get("contrast_threshold", 0.2)
    stats = learned_event_statistics(counts, rel, event["observability"], bins, contrast,
                                     kappa=float(theory_cfg.get("support_kappa", 1.0)))
    delta = log_intensity_delta(pred, float(theory_cfg.get("intensity_eps", 1e-4)))
    residual = delta - stats["learned_delta"]
    # Stop gradients through loss weights, but retain R gradients in D_R (Eq. 33).
    losses["event_physics"] = (stats["learned_support"].detach()
                               * charbonnier(residual, charbonnier_eps)).mean()
    quiet_weight = (stats["interval_observability"] * (1.0 - stats["learned_support"])).detach()
    quiet_penalty = (delta.abs() - float(theory_cfg.get("quiet_margin", 0.02))).clamp_min(0.0)
    losses["quiet"] = (quiet_weight * charbonnier(quiet_penalty, charbonnier_eps)).mean()

    with torch.no_grad():
        priors = count_priors(counts, bins)
        count_delta = signed_event_delta(counts, priors["count_support"], bins, contrast)
        target = torch.exp(-(delta.detach() - count_delta).abs())
    mask = stats["event_mask"]
    losses["reliability_calibration"] = (mask * (stats["supported_confidence"] - target).square()).sum() \
        / (mask.sum() + 1e-6)
    losses["reliability_prior"] = F.mse_loss(rel, priors["count_support"])
    losses["reliability_supervision"] = 0.02 * losses["reliability_calibration"] \
        + 0.01 * losses["reliability_prior"]
    total = zero
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    losses["total"] = total
    return total, losses
