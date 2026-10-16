"""Count-domain quantities from Eqs. (16), (17), and (23)--(25)."""
from __future__ import annotations

import torch


def interval_view(value: torch.Tensor, event_bins: int, clip_len: int | None = None) -> torch.Tensor:
    if value.ndim != 5 or event_bins < 1 or value.shape[2] % event_bins:
        raise ValueError("Expected [B,C,(T-1)*event_bins,H,W] without temporal resampling.")
    b, c, steps, h, w = value.shape
    if clip_len is not None and steps != (clip_len - 1) * event_bins:
        raise ValueError("Event-bin count does not match the requested clip length.")
    return value.reshape(b, c, steps // event_bins, event_bins, h, w)


def count_priors(counts: torch.Tensor, event_bins: int, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """S^{p,h}, Q^h, pi^h and q^h, with percentiles per sample/interval."""
    if counts.shape[1] != 2 or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("event_counts must contain finite nonnegative positive/negative counts.")
    counts = counts.float()
    bins = interval_view(counts, event_bins)
    total = bins.sum(dim=1, keepdim=True)
    support = torch.zeros_like(bins)
    silence = torch.ones_like(total)
    for sample in range(bins.shape[0]):
        for interval in range(bins.shape[2]):
            n = total[sample, :, interval]
            active = n[n > 0]
            if not active.numel():
                continue
            p95 = torch.quantile(active, 0.95).clamp_min(1.0)
            p99 = torch.maximum(torch.quantile(active, 0.99), p95)
            activity = (torch.log1p(n) / torch.log1p(p95)).clamp(0.0, 1.0)
            hot = torch.exp(-(n - p99).clamp_min(0.0) / (p99 + eps))
            polarity_counts = bins[sample, :, interval]
            support[sample, :, interval] = (
                (polarity_counts > 0) * activity * hot
                * (0.25 + 0.75 * polarity_counts / (n + eps))
            )
            silence[sample, :, interval] = hot * torch.exp(-n / (p95 + eps))
    precision = (bins * support).sum(dim=(1, 3)).unsqueeze(1) / (
        bins.sum(dim=(1, 3)).unsqueeze(1) + eps
    )
    quiet = silence.mean(dim=3)
    return {
        "count_support": support.reshape_as(counts),
        "count_silence": silence.reshape(counts.shape[0], 1, counts.shape[2], *counts.shape[-2:]),
        "count_precision": precision,
        "count_quiet": quiet,
        "count_motion": (1.0 - quiet) * (0.25 + 0.75 * precision),
    }


def signed_event_delta(counts: torch.Tensor, confidence: torch.Tensor, event_bins: int,
                       contrast: float | torch.Tensor = 0.2) -> torch.Tensor:
    weighted = interval_view(counts.float() * confidence.float(), event_bins)
    # Contrast is scalar, a batch of scalars, or polarity-specific [B,2].
    cp = torch.as_tensor(contrast, device=counts.device, dtype=torch.float32)
    if cp.ndim == 0:
        cp = cp.reshape(1, 1, 1, 1, 1, 1)
    elif cp.ndim == 1:
        if cp.numel() != counts.shape[0]:
            raise ValueError("Use [B,2] for polarity-specific contrast thresholds.")
        cp = cp.reshape(-1, 1, 1, 1, 1, 1)
    elif cp.shape == (counts.shape[0], 2):
        cp = cp.reshape(-1, 2, 1, 1, 1, 1)
    else:
        raise ValueError("Contrast threshold must be scalar, [B], or [B,2].")
    weighted = weighted * cp
    return (weighted[:, :1] - weighted[:, 1:2]).sum(dim=3)


def learned_event_statistics(counts: torch.Tensor, confidence: torch.Tensor,
                             observability: torch.Tensor, event_bins: int,
                             contrast: float | torch.Tensor = 0.2,
                             kappa: float = 1.0, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    if kappa <= 0 or confidence.shape != counts.shape:
        raise ValueError("kappa must be positive and confidence must match raw count voxels.")
    bins = interval_view(counts.float(), event_bins)
    rel = interval_view(confidence.float(), event_bins)
    mass = (bins * rel).sum(dim=(1, 3)).unsqueeze(1)
    bin_mass = bins.sum(dim=1, keepdim=True)
    active_bins = (bin_mass > 0).to(bins.dtype)
    bin_confidence = (bins * rel).sum(dim=1, keepdim=True) / (bin_mass + eps)
    supported_confidence = (active_bins * bin_confidence).sum(dim=3) / (active_bins.sum(dim=3) + eps)
    return {
        "learned_mass": mass,
        "learned_support": mass / (mass + kappa),
        "learned_delta": signed_event_delta(counts, confidence, event_bins, contrast),
        "interval_observability": interval_view(observability.float(), event_bins).mean(dim=3),
        "supported_confidence": supported_confidence,
        "event_mask": (bin_mass.sum(dim=3) > 0).to(bins.dtype),
    }


def event_confidence_at_events(confidence: torch.Tensor, xs, ys, ts, ps,
                                frame_timestamps, event_bins: int) -> torch.Tensor:
    """Eq. (10) for one clip: maps [2,(T-1)*Be,H,W] to scores in event order."""
    if confidence.ndim != 4 or confidence.shape[0] != 2:
        raise ValueError("Expected one clip of polarity confidence maps.")
    device = confidence.device
    x, y = (torch.as_tensor(a, device=device).long() for a in (xs, ys))
    times = torch.as_tensor(ts, device=device, dtype=torch.float64)
    boundaries = torch.as_tensor(frame_timestamps, device=device, dtype=torch.float64)
    polarity = torch.as_tensor(ps, device=device)
    if not (x.shape == y.shape == times.shape == polarity.shape):
        raise ValueError("Event coordinates, timestamps, and polarity must have equal shapes.")
    if boundaries.ndim != 1 or len(boundaries) < 2 or (boundaries[1:] <= boundaries[:-1]).any():
        raise ValueError("Frame timestamps must be strictly increasing.")
    if confidence.shape[1] != (len(boundaries) - 1) * event_bins or event_bins < 1:
        raise ValueError("Confidence bins do not match interpolation intervals.")
    valid = (torch.isfinite(times) & (times >= boundaries[0]) & (times < boundaries[-1])
             & (x >= 0) & (x < confidence.shape[-1]) & (y >= 0) & (y < confidence.shape[-2]))
    if not valid.all():
        raise ValueError("Events must lie within the spatial field and half-open anchor interval.")
    interval = torch.bucketize(times.contiguous(), boundaries, right=True) - 1
    tau = (event_bins - 1) * (times - boundaries[interval]) / (boundaries[interval + 1] - boundaries[interval])
    lower = tau.floor().long()
    upper = (lower + 1).clamp_max(event_bins - 1)
    weight = (tau - lower).to(confidence)
    channel = (polarity <= 0).long()
    offset = interval * event_bins
    return ((1 - weight) * confidence[channel, offset + lower, y, x]
            + weight * confidence[channel, offset + upper, y, x])
