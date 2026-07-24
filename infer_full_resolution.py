from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# PyCharm default: full-resolution tiled inference (1120 x 660)
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(
    os.environ.get("RAE_CONFIG", PROJECT_DIR / "configs" / "inference.yaml")
)
TEST_H5_PATH = Path(os.environ.get("RAE_TEST_H5", PROJECT_DIR / "data" / "sample.h5"))
RAFT_WEIGHTS_PATH = Path(
    os.environ.get("RAE_RAFT_WEIGHTS", PROJECT_DIR / "raft_small_C_T_V2-01064c6d.pth")
)
OUT_DIR_ROOT = PROJECT_DIR / "outputs"

CHECKPOINT_CANDIDATES = [
    PROJECT_DIR / "FINAL_BEST_INFERENCE.pt",
    PROJECT_DIR / "best_occlusion_refinement.pt",
]

DEFAULT_MODE = "tiled"
NUM_SAMPLES = 999999
CLIP_STRIDE = 6
KEYFRAME_POLICY = "endpoints"
KEYFRAME_STRIDE = 6

TILE_SIZE = 256
MIN_OVERLAP = 64
PAD_MULTIPLE = 16
GIF_DURATION = 0.08


def add_src_to_path() -> None:
    src = PROJECT_DIR / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def resolve_checkpoint() -> Path:
    override = os.environ.get("RAE_CHECKPOINT")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"RAE_CHECKPOINT does not exist: {path}")
        return path
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Checkpoint not found. Tried:\n"
        + "\n".join(f"  {path}" for path in CHECKPOINT_CANDIDATES)
    )


def load_runtime(device: torch.device):
    add_src_to_path()
    import infer_local as base
    from raevdm.flow_warp import load_frozen_raft_small
    from raevdm.models import build_model
    from raevdm.utils import load_config

    checkpoint_path = resolve_checkpoint()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = load_config(CONFIG_PATH)
    checkpoint_config = checkpoint.get("config", {})
    for section in (
        "model",
        "diffusion",
        "finetune",
        "residual_correction",
        "flow_fusion",
        "occlusion_refinement",
        "theory",
        "residual",
    ):
        if section in checkpoint_config:
            config[section] = checkpoint_config[section]

    if not base.is_occlusion_refinement_checkpoint(config):
        raise RuntimeError("Full-resolution script requires occlusion-refinement weights.")

    model = build_model(config).to(device)
    state_key = "model_ema" if "model_ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.eval()
    raft, raft_transforms = load_frozen_raft_small(
        device,
        RAFT_WEIGHTS_PATH,
    )
    residual_params = base.residual_params_from_config(config)
    flow_fusion_params = dict(config.get("flow_fusion", {}) or {})
    return (
        base,
        config,
        checkpoint,
        checkpoint_path,
        model,
        raft,
        raft_transforms,
        residual_params,
        flow_fusion_params,
    )


def build_dataset(config: dict[str, Any]):
    from raevdm.data import RLEDClipDataset

    data = dict(config["data"])
    data.update(
        {
            "root": str(TEST_H5_PATH.parent),
            "file_pattern": TEST_H5_PATH.name,
            "split": "all",
            "max_files": 1,
            "random_crop": False,
            "random_flip": False,
            "clip_stride": CLIP_STRIDE,
            "crop_size": None,
            "keyframe_policy": KEYFRAME_POLICY,
            "keyframe_stride": KEYFRAME_STRIDE,
        }
    )
    return RLEDClipDataset(**data)


def spatial_slice(batch: dict[str, Any], top: int, left: int, size: int):
    height, width = batch["frames"].shape[-2:]
    sliced = {}
    for key, value in batch.items():
        if (
            torch.is_tensor(value)
            and value.ndim >= 4
            and tuple(value.shape[-2:]) == (height, width)
        ):
            sliced[key] = value[..., top : top + size, left : left + size]
        else:
            sliced[key] = value
    return sliced


def pad_spatial_batch(batch: dict[str, Any], multiple: int):
    height, width = batch["frames"].shape[-2:]
    padded_h = math.ceil(height / multiple) * multiple
    padded_w = math.ceil(width / multiple) * multiple
    pad_bottom = padded_h - height
    pad_right = padded_w - width
    padded = {}
    for key, value in batch.items():
        if (
            torch.is_tensor(value)
            and value.ndim >= 4
            and tuple(value.shape[-2:]) == (height, width)
        ):
            leading_shape = value.shape[:-3]
            channels = value.shape[-3]
            flattened = value.reshape(-1, channels, height, width)
            padded_value = F.pad(
                flattened,
                (0, pad_right, 0, pad_bottom),
                mode="replicate",
            )
            padded[key] = padded_value.reshape(
                *leading_shape,
                channels,
                padded_h,
                padded_w,
            )
        else:
            padded[key] = value
    return padded, height, width


def axis_positions(length: int, tile_size: int, min_overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - min_overlap
    count = math.ceil((length - tile_size) / stride) + 1
    return [
        int(round(index * (length - tile_size) / (count - 1)))
        for index in range(count)
    ]


def axis_blend_weights(
    positions: list[int],
    index: int,
    tile_size: int,
    device: torch.device,
) -> torch.Tensor:
    weight = torch.ones(tile_size, device=device, dtype=torch.float32)
    start = positions[index]
    if index > 0:
        left_overlap = positions[index - 1] + tile_size - start
        phase = (torch.arange(left_overlap, device=device) + 0.5) / left_overlap
        weight[:left_overlap] *= 0.5 - 0.5 * torch.cos(math.pi * phase)
    if index + 1 < len(positions):
        right_overlap = start + tile_size - positions[index + 1]
        phase = (torch.arange(right_overlap, device=device) + 0.5) / right_overlap
        right_ramp = 0.5 + 0.5 * torch.cos(math.pi * phase)
        weight[-right_overlap:] *= right_ramp
    return weight


def direct_predict(
    base,
    model,
    raft,
    raft_transforms,
    batch,
    residual_params,
    flow_fusion_params,
    device,
) -> tuple[torch.Tensor, float, float]:
    from raevdm.utils import to_device

    padded, height, width = pad_spatial_batch(batch, PAD_MULTIPLE)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    padded = to_device(padded, device)
    with torch.no_grad():
        pred = base.sample_flow_fusion(
            model,
            raft,
            raft_transforms,
            padded,
            residual_params,
            flow_fusion_params,
        )
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    return pred[..., :height, :width].cpu(), elapsed, peak_gib


def tiled_predict(
    base,
    model,
    raft,
    raft_transforms,
    batch,
    residual_params,
    flow_fusion_params,
    device,
) -> tuple[torch.Tensor, float, float, tuple[int, int]]:
    from raevdm.utils import to_device

    height, width = batch["frames"].shape[-2:]
    if height < TILE_SIZE or width < TILE_SIZE:
        raise ValueError(
            f"Frame {width}x{height} is smaller than tile size {TILE_SIZE}."
        )
    ys = axis_positions(height, TILE_SIZE, MIN_OVERLAP)
    xs = axis_positions(width, TILE_SIZE, MIN_OVERLAP)
    y_weights = [
        axis_blend_weights(ys, index, TILE_SIZE, torch.device("cpu"))
        for index in range(len(ys))
    ]
    x_weights = [
        axis_blend_weights(xs, index, TILE_SIZE, torch.device("cpu"))
        for index in range(len(xs))
    ]

    frames_t = batch["frames"].shape[1]
    channels = batch["frames"].shape[2]
    accumulated = torch.zeros(
        (frames_t, channels, height, width),
        dtype=torch.float32,
    )
    weight_sum = torch.zeros((1, 1, height, width), dtype=torch.float32)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    tile_bar = tqdm(
        total=len(ys) * len(xs),
        desc="tiles",
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for y_index, top in enumerate(ys):
            for x_index, left in enumerate(xs):
                tile_batch = spatial_slice(batch, top, left, TILE_SIZE)
                tile_batch = to_device(tile_batch, device)
                tile_pred = base.sample_flow_fusion(
                    model,
                    raft,
                    raft_transforms,
                    tile_batch,
                    residual_params,
                    flow_fusion_params,
                )[0].float().cpu()
                window = (
                    y_weights[y_index][:, None]
                    * x_weights[x_index][None, :]
                )[None, None]
                accumulated[
                    ...,
                    top : top + TILE_SIZE,
                    left : left + TILE_SIZE,
                ] += tile_pred * window
                weight_sum[
                    ...,
                    top : top + TILE_SIZE,
                    left : left + TILE_SIZE,
                ] += window
                del tile_batch, tile_pred
                tile_bar.update(1)
    tile_bar.close()
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    pred = (accumulated / weight_sum.clamp_min(1e-6)).clamp(0.0, 1.0)
    return pred.unsqueeze(0), elapsed, peak_gib, (len(ys), len(xs))


def frames_to_uint8(video: torch.Tensor) -> list[np.ndarray]:
    frames = (video.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return [frame[0] if frame.ndim == 3 else frame for frame in frames]


def save_clip(video: torch.Tensor, sample_dir: Path, name: str) -> None:
    frame_dir = sample_dir / name
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = frames_to_uint8(video)
    for index, frame in enumerate(frames):
        imageio.imwrite(frame_dir / f"{index:03d}.png", frame)
    imageio.mimsave(
        sample_dir / f"{name}.gif",
        frames,
        duration=GIF_DURATION,
    )


def batch_start_index(batch: dict[str, Any]) -> int:
    value = batch.get("start_index", 0)
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def add_timeline(
    timeline: dict[int, list[np.ndarray]],
    start: int,
    frames: list[np.ndarray],
) -> None:
    for offset, frame in enumerate(frames):
        timeline.setdefault(start + offset, []).append(frame)


def timeline_frames(timeline: dict[int, list[np.ndarray]]) -> list[np.ndarray]:
    return [
        np.mean(np.stack(timeline[index], axis=0), axis=0)
        .round()
        .clip(0, 255)
        .astype(np.uint8)
        for index in sorted(timeline)
    ]


def intermediate_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    pred_middle = pred[:, 1:-1].float()
    gt_middle = gt[:, 1:-1].float()
    error = pred_middle - gt_middle
    mae = float(error.abs().mean())
    mse = float(error.square().mean())
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    pred_delta = pred[:, 2:-1] - pred[:, 1:-2]
    gt_delta = gt[:, 2:-1] - gt[:, 1:-2]
    temporal = float((pred_delta - gt_delta).abs().mean())
    return {"mae": mae, "psnr": psnr, "temporal_mae": temporal}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("direct", "tiled"),
        default=DEFAULT_MODE,
    )
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-resolution inference.")
    device = torch.device("cuda")
    (
        base,
        config,
        checkpoint,
        checkpoint_path,
        model,
        raft,
        raft_transforms,
        residual_params,
        flow_fusion_params,
    ) = load_runtime(device)
    dataset = build_dataset(config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    out_dir = OUT_DIR_ROOT / (
        f"FULL_RES_{args.mode.upper()}_{checkpoint_path.stem}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch/step: {checkpoint.get('epoch')}/{checkpoint.get('step')}")
    print(f"Original resolution: 1120x660")
    if args.mode == "tiled":
        print(f"Tile: {TILE_SIZE}x{TILE_SIZE}, minimum overlap: {MIN_OVERLAP}")
    print(f"Output: {out_dir}")

    stitched_gt: dict[int, list[np.ndarray]] = {}
    stitched_pred: dict[int, list[np.ndarray]] = {}
    metric_rows = []
    for sample_index, batch in enumerate(loader):
        if sample_index >= args.num_samples:
            break
        try:
            if args.mode == "direct":
                pred, elapsed, peak_gib = direct_predict(
                    base,
                    model,
                    raft,
                    raft_transforms,
                    batch,
                    residual_params,
                    flow_fusion_params,
                    device,
                )
                tile_grid = None
            else:
                pred, elapsed, peak_gib, tile_grid = tiled_predict(
                    base,
                    model,
                    raft,
                    raft_transforms,
                    batch,
                    residual_params,
                    flow_fusion_params,
                    device,
                )
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            print(f"DIRECT_OOM: {exc}")
            raise

        gt = batch["frames"].float()
        metrics = intermediate_metrics(pred, gt)
        metric_rows.append(metrics)
        grid_text = f", grid={tile_grid[0]}x{tile_grid[1]}" if tile_grid else ""
        print(
            f"sample {sample_index:03d}: "
            f"PSNR={metrics['psnr']:.4f}, MAE={metrics['mae']:.6f}, "
            f"temporal={metrics['temporal_mae']:.6f}, "
            f"time={elapsed:.1f}s, peak={peak_gib:.2f}GiB{grid_text}"
        )

        sample_dir = out_dir / f"sample_{sample_index:03d}"
        save_clip(gt[0], sample_dir, "ground_truth")
        save_clip(pred[0], sample_dir, "prediction")
        start = batch_start_index(batch)
        add_timeline(stitched_gt, start, frames_to_uint8(gt[0]))
        add_timeline(stitched_pred, start, frames_to_uint8(pred[0]))

    if metric_rows:
        mean_metrics = {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in metric_rows[0]
        }
        print(f"Mean metrics: {mean_metrics}")
        imageio.mimsave(
            out_dir / "ground_truth_full.gif",
            timeline_frames(stitched_gt),
            duration=GIF_DURATION,
        )
        imageio.mimsave(
            out_dir / "prediction_full.gif",
            timeline_frames(stitched_pred),
            duration=GIF_DURATION,
        )
        print(f"Ground truth: {out_dir / 'ground_truth_full.gif'}")
        print(f"Prediction: {out_dir / 'prediction_full.gif'}")


if __name__ == "__main__":
    main()
