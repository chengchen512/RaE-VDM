from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class EventRepresentation:
    condition: np.ndarray
    raw: np.ndarray
    reliability: np.ndarray
    observability: np.ndarray
    event_delta: np.ndarray
    interval_precision: np.ndarray
    interval_observability: np.ndarray
    counts: np.ndarray


def normalize_polarity(ps: np.ndarray) -> np.ndarray:
    ps = np.asarray(ps)
    if ps.size == 0:
        return ps.astype(np.float32)
    if np.nanmin(ps) >= 0:
        return np.where(ps > 0, 1.0, -1.0).astype(np.float32)
    return np.where(ps > 0, 1.0, -1.0).astype(np.float32)


def robust_count_normalize(voxel: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    if voxel.size == 0:
        return voxel.astype(np.float32)
    scale = np.percentile(np.abs(voxel), percentile)
    scale = max(float(scale), 1.0)
    return np.clip(voxel / scale, 0.0, 1.0).astype(np.float32)


def events_to_neg_pos_voxels(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    ps: np.ndarray,
    height: int,
    width: int,
    bins: int,
    t0: float | None = None,
    t1: float | None = None,
    temporal_bilinear: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate events into positive and negative voxel grids.

    Output shape is [bins, height, width] for each polarity. Values are counts,
    not signed intensities. Temporal bilinear interpolation follows the common
    event voxel construction used in event-based reconstruction literature.
    """
    bins = int(max(bins, 1))
    pos = np.zeros((bins, height, width), dtype=np.float32)
    neg = np.zeros((bins, height, width), dtype=np.float32)
    if len(xs) == 0:
        return pos, neg

    xs = np.asarray(xs).astype(np.int64)
    ys = np.asarray(ys).astype(np.int64)
    ts = np.asarray(ts).astype(np.float64)
    ps = normalize_polarity(np.asarray(ps))

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height) & np.isfinite(ts)
    if t0 is not None:
        valid &= ts >= float(t0)
    if t1 is not None:
        valid &= ts < float(t1)
    if not np.any(valid):
        return pos, neg
    xs, ys, ts, ps = xs[valid], ys[valid], ts[valid], ps[valid]

    if t0 is None:
        t0 = float(ts[0])
    if t1 is None:
        t1 = float(ts[-1])
    duration = max(float(t1) - float(t0), 1e-6)

    if bins == 1:
        tau = np.zeros_like(ts, dtype=np.float32)
    else:
        tau = ((ts - float(t0)) / duration * (bins - 1)).astype(np.float32)
        tau = np.clip(tau, 0.0, bins - 1.0)

    if temporal_bilinear and bins > 1:
        lower = np.floor(tau).astype(np.int64)
        upper = np.clip(lower + 1, 0, bins - 1)
        w_upper = tau - lower.astype(np.float32)
        w_lower = 1.0 - w_upper
        for b, w in ((lower, w_lower), (upper, w_upper)):
            mask_pos = ps > 0
            mask_neg = ~mask_pos
            np.add.at(pos, (b[mask_pos], ys[mask_pos], xs[mask_pos]), w[mask_pos])
            np.add.at(neg, (b[mask_neg], ys[mask_neg], xs[mask_neg]), w[mask_neg])
    else:
        b = np.clip(np.round(tau).astype(np.int64), 0, bins - 1)
        mask_pos = ps > 0
        mask_neg = ~mask_pos
        np.add.at(pos, (b[mask_pos], ys[mask_pos], xs[mask_pos]), 1.0)
        np.add.at(neg, (b[mask_neg], ys[mask_neg], xs[mask_neg]), 1.0)

    return pos, neg


def reliability_prior_from_counts(pos: np.ndarray, neg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a conservative reliability prior from local event statistics.

    The prior is not used as ground truth. It gives the reliability head a
    stabilizing target in early training and remains removable for ablations.
    """
    total = pos + neg
    if not np.any(total > 0):
        zero_rel = np.zeros_like(total, dtype=np.float32)
        one_obs = np.ones_like(total, dtype=np.float32)
        return zero_rel, zero_rel.copy(), one_obs

    active = total[total > 0]
    p95 = max(float(np.percentile(active, 95.0)), 1.0)
    p99 = max(float(np.percentile(active, 99.0)), p95)

    activity = np.clip(np.log1p(total) / np.log1p(p95), 0.0, 1.0)
    hot_penalty = np.exp(-np.maximum(total - p99, 0.0) / (p99 + 1e-6))
    purity_pos = pos / (total + 1e-6)
    purity_neg = neg / (total + 1e-6)

    rel_pos = activity * hot_penalty * (0.25 + 0.75 * purity_pos)
    rel_neg = activity * hot_penalty * (0.25 + 0.75 * purity_neg)
    rel_pos = np.where(pos > 0, rel_pos, 0.0)
    rel_neg = np.where(neg > 0, rel_neg, 0.0)

    observability = hot_penalty * np.exp(-total / (p95 + 1e-6))
    observability = np.clip(observability, 0.0, 1.0)
    return rel_pos.astype(np.float32), rel_neg.astype(np.float32), observability.astype(np.float32)


def build_event_representation(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    ps: np.ndarray,
    height: int,
    width: int,
    bins: int,
    t0: float,
    t1: float,
    contrast_threshold: float | tuple[float, float] = 0.2,
) -> EventRepresentation:
    pos_counts, neg_counts = events_to_neg_pos_voxels(xs, ys, ts, ps, height, width, bins, t0=t0, t1=t1)
    rel_pos, rel_neg, observability = reliability_prior_from_counts(pos_counts, neg_counts)

    pos_norm = robust_count_normalize(pos_counts)
    neg_norm = robust_count_normalize(neg_counts)
    weighted_pos = pos_norm * rel_pos
    weighted_neg = neg_norm * rel_neg

    condition = np.stack(
        [pos_norm, neg_norm, rel_pos, rel_neg, weighted_pos, weighted_neg, observability],
        axis=0,
    ).astype(np.float32)
    raw = np.stack([pos_norm, neg_norm], axis=0).astype(np.float32)
    counts = np.stack([pos_counts, neg_counts], axis=0).astype(np.float32)
    reliability = np.stack([rel_pos, rel_neg], axis=0).astype(np.float32)
    observability = observability[None].astype(np.float32)

    contrast = np.asarray(contrast_threshold, dtype=np.float32)
    if contrast.ndim == 0:
        cp, cn = float(contrast), float(contrast)
    elif contrast.shape == (2,):
        cp, cn = contrast
    else:
        raise ValueError("contrast_threshold must be a scalar or [positive, negative].")
    delta = (cp * pos_counts * rel_pos - cn * neg_counts * rel_neg).sum(axis=0)
    delta = delta.astype(np.float32)

    total_counts = pos_counts + neg_counts
    rel_weighted = (rel_pos * pos_counts + rel_neg * neg_counts).sum(axis=0)
    precision = rel_weighted / (total_counts.sum(axis=0) + 1e-6)
    precision = np.clip(precision, 0.0, 1.0).astype(np.float32)
    interval_obs = observability.mean(axis=1)[0].astype(np.float32)

    return EventRepresentation(
        condition=condition,
        raw=raw,
        reliability=reliability,
        observability=observability,
        event_delta=delta,
        interval_precision=precision,
        interval_observability=interval_obs,
        counts=counts,
    )
