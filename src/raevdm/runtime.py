"""Deterministic inference shared by CLI, evaluation, and tiled inference."""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from .event_statistics import count_priors
from .flow_warp import load_frozen_raft_small
from .models import build_model
from .stages import geometry_inputs, load_stage_weights
from .utils import load_config, merge_config, to_device


def load_runtime(config_path, checkpoint_path, device, raft_weights=None, use_ema=True):
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved = checkpoint.get("config", {})
    # Architecture dimensions must come from the checkpoint, not an inference preset.
    for section in ("model", "diffusion", "residual", "residual_correction", "flow_fusion"):
        if section in saved:
            config[section] = (dict(saved[section]) if section in {"model", "diffusion"}
                               else merge_config(config.get(section, {}), saved[section]))
    if "residual_correction" in saved and "residual" not in saved:
        config["residual"] = merge_config(config.get("residual", {}), saved["residual_correction"])
    if saved.get("implementation_version") != "tcsvt-20261016":
        warnings.warn("Legacy checkpoint: tensor compatibility does not validate the updated "
                      "count-gate/loss/time-coordinate semantics. Re-evaluate before reporting results.", stacklevel=2)
    config["model"]["event_gate_source"] = "count"
    model = build_model(config).to(device).eval()
    if not use_ema and "model_ema" in checkpoint:
        checkpoint = {key: value for key, value in checkpoint.items() if key != "model_ema"}
    load_stage_weights(model, checkpoint)
    raft = transforms = None
    if model.use_flow_fusion:
        path = raft_weights or config.get("flow_fusion", {}).get("raft_weights")
        if not path:
            raise ValueError("A local RAFT-Small weights path is required for geometry inference.")
        raft, transforms = load_frozen_raft_small(device, path)
    return model, config, raft, transforms


def observation_batch(batch: dict) -> dict:
    """Remove all evaluation targets before invoking reconstruction modules."""
    result = {key: value for key, value in batch.items() if key != "frames"}
    mask = result["keyframe_mask"]
    if mask[:, 1:-1].any() or not (mask[:, 0] == 1).all() or not (mask[:, -1] == 1).all():
        raise ValueError("Paper inference requires exactly the two endpoint anchors.")
    return result


@torch.no_grad()
def predict(model, batch: dict, config: dict, raft=None, transforms=None):
    observed = observation_batch(batch)
    flow = geometry_inputs(observed, raft, transforms,
                           int(config.get("flow_fusion", {}).get("flow_updates_val", 12))) \
        if model.use_flow_fusion else None
    residual = config.get("residual", config.get("residual_correction", {}))
    return model.reconstruct(observed, flow, residual,
                             float(config.get("flow_fusion", {}).get("base_raft_peak", 0.55)))


def axis_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size < 1 or not 0 <= overlap < tile_size:
        raise ValueError("Require 0 <= overlap < tile_size.")
    if length <= tile_size:
        return [0]
    count = math.ceil((length - tile_size) / (tile_size - overlap)) + 1
    return [round(index * (length - tile_size) / (count - 1)) for index in range(count)]


def axis_blend_weights(positions, index, tile_size):
    weight = torch.ones(tile_size)
    start = positions[index]
    if index:
        overlap = positions[index - 1] + tile_size - start
        phase = (torch.arange(overlap) + 0.5) / overlap
        weight[:overlap] *= 0.5 - 0.5 * torch.cos(math.pi * phase)
    if index + 1 < len(positions):
        overlap = start + tile_size - positions[index + 1]
        phase = (torch.arange(overlap) + 0.5) / overlap
        weight[-overlap:] *= 0.5 + 0.5 * torch.cos(math.pi * phase)
    return weight


@torch.no_grad()
def predict_tiled(model, batch, config, device, raft=None, transforms=None,
                  tile_size=256, overlap=64):
    """Native field of view, raised-cosine blending, no input resizing."""
    batch = observation_batch(batch)
    batch = to_device(batch, torch.device("cpu"))
    anchors = batch["keyframes"]
    height, width = anchors.shape[-2:]
    bins_value = batch.get("event_bins", config["data"]["event_bins"])
    bins = int(bins_value.flatten()[0]) if torch.is_tensor(bins_value) else int(bins_value)
    # Compute interval percentiles on the full field before making spatial tiles.
    batch["count_gate_motion"] = count_priors(batch["event_counts"], bins)["count_motion"]
    ys, xs = axis_positions(height, tile_size, overlap), axis_positions(width, tile_size, overlap)
    accumulated = torch.zeros_like(anchors, device="cpu", dtype=torch.float32)
    denominator = torch.zeros(1, 1, 1, height, width)
    multiple = 2 ** (len(config["model"].get("channel_mult", [1, 2, 4])) - 1)
    for yi, top in enumerate(ys):
        for xi, left in enumerate(xs):
            bottom, right = min(top + tile_size, height), min(left + tile_size, width)
            tile = {}
            pad_h = (-int(bottom - top)) % multiple
            pad_w = (-int(right - left)) % multiple
            for key, value in batch.items():
                if torch.is_tensor(value) and value.ndim >= 4 and value.shape[-2:] == (height, width):
                    value = value[..., top:bottom, left:right]
                    shape = value.shape
                    value = F.pad(value.reshape(-1, 1, *shape[-2:]), (0, pad_w, 0, pad_h), mode="replicate")
                    value = value.reshape(*shape[:-2], bottom - top + pad_h, right - left + pad_w)
                tile[key] = value
            result = predict(model, to_device(tile, device), config, raft, transforms).float().cpu()
            result = result[..., :bottom - top, :right - left]
            wy = axis_blend_weights(ys, yi, min(tile_size, height))
            wx = axis_blend_weights(xs, xi, min(tile_size, width))
            window = (wy[:, None] * wx[None, :])[None, None, None]
            accumulated[..., top:bottom, left:right] += result * window
            denominator[..., top:bottom, left:right] += window
    result = (accumulated / denominator).clamp(0, 1)
    mask = batch["keyframe_mask"]
    return result * (1 - mask) + anchors * mask
