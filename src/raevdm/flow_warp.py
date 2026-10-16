from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def load_frozen_raft_small(
    device: torch.device,
    weights_path: str | Path | None = None,
) -> tuple[nn.Module, object]:
    try:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    except Exception as exc:
        raise RuntimeError(
            "torchvision optical-flow models are unavailable. "
            "Install a torchvision version compatible with the current PyTorch."
        ) from exc

    weights = Raft_Small_Weights.DEFAULT
    path = Path(weights_path) if weights_path else None
    if path is not None and path.exists():
        model = raft_small(weights=None, progress=False)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    elif path is not None:
        raise FileNotFoundError(f"RAFT weights do not exist: {path}")
    else:
        model = raft_small(weights=weights, progress=True)

    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms()


def _pad_to_multiple_of_8(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (pad_h, pad_w)


@torch.no_grad()
def estimate_bidirectional_flow(
    raft: nn.Module,
    transforms: object,
    image0: torch.Tensor,
    image1: torch.Tensor,
    flow_updates: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb0 = image0.repeat(1, 3, 1, 1)
    rgb1 = image1.repeat(1, 3, 1, 1)
    rgb0, padding = _pad_to_multiple_of_8(rgb0)
    rgb1, _ = _pad_to_multiple_of_8(rgb1)
    input0, input1 = transforms(rgb0, rgb1)

    flow01 = raft(input0, input1, num_flow_updates=int(flow_updates))[-1]
    flow10 = raft(input1, input0, num_flow_updates=int(flow_updates))[-1]
    pad_h, pad_w = padding
    if pad_h:
        flow01 = flow01[..., :-pad_h, :]
        flow10 = flow10[..., :-pad_h, :]
    if pad_w:
        flow01 = flow01[..., :, :-pad_w]
        flow10 = flow10[..., :, :-pad_w]
    return flow01, flow10


def backward_warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, _, h, w = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    x = xx[None] + flow[:, 0]
    y = yy[None] + flow[:, 1]
    grid_x = 2.0 * x / max(w - 1, 1) - 1.0
    grid_y = 2.0 * y / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


@torch.no_grad()
def flow_endpoint_confidence(
    image0: torch.Tensor,
    image1: torch.Tensor,
    flow01: torch.Tensor,
    flow10: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    image1_at_0 = backward_warp(image1, flow01)
    image0_at_1 = backward_warp(image0, flow10)
    photo0 = (image0 - image1_at_0).abs()
    photo1 = (image1 - image0_at_1).abs()

    flow10_at_0 = backward_warp(flow10, flow01)
    flow01_at_1 = backward_warp(flow01, flow10)
    cycle0 = torch.linalg.vector_norm(flow01 + flow10_at_0, dim=1, keepdim=True)
    cycle1 = torch.linalg.vector_norm(flow10 + flow01_at_1, dim=1, keepdim=True)
    conf0 = torch.exp(-photo0 / 0.10) * torch.exp(-cycle0 / 5.0)
    conf1 = torch.exp(-photo1 / 0.10) * torch.exp(-cycle1 / 5.0)
    return conf0.clamp_min(0.02), conf1.clamp_min(0.02)


def forward_splat(
    image: torch.Tensor,
    flow: torch.Tensor,
    scale: float | torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, c, h, w = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    if torch.is_tensor(scale):
        scale = scale.to(flow).reshape(b, 1, 1)
    dst_x = xx[None] + scale * flow[:, 0]
    dst_y = yy[None] + scale * flow[:, 1]
    x0 = torch.floor(dst_x)
    y0 = torch.floor(dst_y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    value_sum = image.new_zeros((b, c, h * w))
    weight_sum = image.new_zeros((b, 1, h * w))
    source = image.flatten(2)
    source_conf = confidence.flatten(2)

    for x, y, weight in (
        (x0, y0, (x1 - dst_x) * (y1 - dst_y)),
        (x1, y0, (dst_x - x0) * (y1 - dst_y)),
        (x0, y1, (x1 - dst_x) * (dst_y - y0)),
        (x1, y1, (dst_x - x0) * (dst_y - y0)),
    ):
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        index = (y.clamp(0, h - 1).long() * w + x.clamp(0, w - 1).long()).flatten(1)
        splat_weight = (weight * valid).flatten(1)[:, None] * source_conf
        value_sum.scatter_add_(2, index[:, None].expand(-1, c, -1), source * splat_weight)
        weight_sum.scatter_add_(2, index[:, None], splat_weight)

    return value_sum.view(b, c, h, w), weight_sum.view(b, 1, h, w)


def _flow_boundary_strength(flow: torch.Tensor) -> torch.Tensor:
    dx = F.pad(flow[..., :, 1:] - flow[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(flow[..., 1:, :] - flow[..., :-1, :], (0, 0, 0, 1))
    strength = torch.sqrt((dx * dx + dy * dy).sum(dim=1, keepdim=True) + 1e-6)
    return (1.0 - torch.exp(-strength / 2.0)).clamp(0.0, 1.0)


@torch.no_grad()
def synthesize_bidirectional_warp_video(
    image0: torch.Tensor,
    image1: torch.Tensor,
    flow01: torch.Tensor,
    flow10: torch.Tensor,
    frames_t: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conf0, conf1 = flow_endpoint_confidence(image0, image1, flow01, flow10)
    warped_frames = []
    confidence_frames = []
    time_frames = []

    for frame_index in range(int(frames_t)):
        t = float(frame_index) / float(max(int(frames_t) - 1, 1))
        if frame_index == 0:
            warped = image0
            confidence = torch.ones_like(image0)
        elif frame_index == int(frames_t) - 1:
            warped = image1
            confidence = torch.ones_like(image1)
        else:
            value0, weight0 = forward_splat(image0, flow01, t, conf0)
            value1, weight1 = forward_splat(image1, flow10, 1.0 - t, conf1)
            weighted_value = (1.0 - t) * value0 + t * value1
            weight = (1.0 - t) * weight0 + t * weight1
            warped = weighted_value / weight.clamp_min(1e-6)
            linear = (1.0 - t) * image0 + t * image1
            warped = torch.where(weight < 0.02, linear, warped).clamp(0.0, 1.0)
            confidence = (1.0 - torch.exp(-weight)).clamp(0.0, 1.0)

        warped_frames.append(warped)
        confidence_frames.append(confidence)
        time_frames.append(torch.ones_like(image0) * t)

    return (
        torch.stack(warped_frames, dim=2),
        torch.stack(confidence_frames, dim=2),
        torch.stack(time_frames, dim=2),
    )


@torch.no_grad()
def synthesize_occlusion_aware_warp_video(
    image0: torch.Tensor,
    image1: torch.Tensor,
    flow01: torch.Tensor,
    flow10: torch.Tensor,
    frames_t: int,
    timestamps: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Endpoint splats, target-domain agreement, visibility, and normalized alpha."""

    conf0, conf1 = flow_endpoint_confidence(image0, image1, flow01, flow10)
    boundary0 = _flow_boundary_strength(flow01)
    boundary1 = _flow_boundary_strength(flow10)

    warped_frames = []
    confidence_frames = []
    time_frames = []
    endpoint0_frames = []
    endpoint1_frames = []
    visibility0_frames = []
    visibility1_frames = []
    occlusion_frames = []
    hole_frames = []
    boundary_frames = []

    total_frames = int(frames_t)
    for frame_index in range(total_frames):
        if timestamps is None:
            t = float(frame_index) / float(max(total_frames - 1, 1))
        else:
            times = timestamps if timestamps.ndim == 2 else timestamps[None]
            t = ((times[:, frame_index] - times[:, 0]) / (times[:, -1] - times[:, 0])).to(image0)
            t = t.reshape(-1, 1, 1, 1)
        if frame_index == 0:
            warped = image0
            confidence = torch.ones_like(image0)
            endpoint0 = image0
            endpoint1 = image0
            visibility0 = torch.ones_like(image0)
            visibility1 = torch.zeros_like(image0)
            occlusion = torch.zeros_like(image0)
            hole = torch.zeros_like(image0)
            boundary = torch.zeros_like(image0)
        elif frame_index == total_frames - 1:
            warped = image1
            confidence = torch.ones_like(image1)
            endpoint0 = image1
            endpoint1 = image1
            visibility0 = torch.zeros_like(image1)
            visibility1 = torch.ones_like(image1)
            occlusion = torch.zeros_like(image1)
            hole = torch.zeros_like(image1)
            boundary = torch.zeros_like(image1)
        else:
            value0, weight0 = forward_splat(image0, flow01, t, conf0)
            value1, weight1 = forward_splat(image1, flow10, 1.0 - t, conf1)
            endpoint0 = value0 / weight0.clamp_min(1e-6)
            endpoint1 = value1 / weight1.clamp_min(1e-6)
            linear = (1.0 - t) * image0 + t * image1
            endpoint0 = torch.where(weight0 < 0.02, linear, endpoint0).clamp(0.0, 1.0)
            endpoint1 = torch.where(weight1 < 0.02, linear, endpoint1).clamp(0.0, 1.0)

            weighted0 = (1.0 - t) * weight0
            weighted1 = t * weight1
            weight = weighted0 + weighted1
            warped = ((1.0 - t) * value0 + t * value1) / weight.clamp_min(1e-6)
            warped = torch.where(weight < 0.02, linear, warped).clamp(0.0, 1.0)
            confidence = ((1.0 - torch.exp(-weight))
                          * torch.exp(-(endpoint0 - endpoint1).abs() / 0.10)).clamp(0.0, 1.0)

            visibility0 = weighted0 / weight.clamp_min(1e-6)
            visibility1 = weighted1 / weight.clamp_min(1e-6)
            both_valid = (weight0 >= 0.02) & (weight1 >= 0.02)
            disagreement = (endpoint0 - endpoint1).abs()
            imbalance = (visibility0 - visibility1).abs()
            occlusion = torch.where(
                both_valid,
                (disagreement * (0.5 + 0.5 * imbalance)).clamp(0.0, 1.0),
                torch.ones_like(disagreement),
            )
            hole = torch.exp(-weight).clamp(0.0, 1.0)

            boundary_value0, boundary_weight0 = forward_splat(boundary0, flow01, t, conf0)
            boundary_value1, boundary_weight1 = forward_splat(boundary1, flow10, 1.0 - t, conf1)
            warped_boundary0 = boundary_value0 / boundary_weight0.clamp_min(1e-6)
            warped_boundary1 = boundary_value1 / boundary_weight1.clamp_min(1e-6)
            boundary = torch.maximum(warped_boundary0, warped_boundary1)
            boundary = torch.maximum(boundary, disagreement).clamp(0.0, 1.0)

        warped_frames.append(warped)
        confidence_frames.append(confidence)
        time_frames.append(torch.ones_like(image0) * t)
        endpoint0_frames.append(endpoint0)
        endpoint1_frames.append(endpoint1)
        visibility0_frames.append(visibility0)
        visibility1_frames.append(visibility1)
        occlusion_frames.append(occlusion)
        hole_frames.append(hole)
        boundary_frames.append(boundary)

    return {
        "warped_video": torch.stack(warped_frames, dim=2),
        "warp_confidence": torch.stack(confidence_frames, dim=2),
        "time_map": torch.stack(time_frames, dim=2),
        "endpoint_warp0": torch.stack(endpoint0_frames, dim=2),
        "endpoint_warp1": torch.stack(endpoint1_frames, dim=2),
        "visibility0": torch.stack(visibility0_frames, dim=2),
        "visibility1": torch.stack(visibility1_frames, dim=2),
        "occlusion_map": torch.stack(occlusion_frames, dim=2),
        "hole_map": torch.stack(hole_frames, dim=2),
        "motion_boundary": torch.stack(boundary_frames, dim=2),
    }
