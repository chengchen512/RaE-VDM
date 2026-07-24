from __future__ import annotations

import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# =========================
# Local no-CLI settings
# =========================

PROJECT_DIR = Path(__file__).resolve().parent

CONFIG_PATH = Path(
    os.environ.get("RAE_CONFIG", PROJECT_DIR / "configs" / "inference.yaml")
)

CHECKPOINT_CANDIDATES = [
    PROJECT_DIR / "FINAL_BEST_INFERENCE.pt",
]
RAFT_WEIGHTS_PATH = Path(
    os.environ.get("RAE_RAFT_WEIGHTS", PROJECT_DIR / "raft_small_C_T_V2-01064c6d.pth")
)

# Local test H5 file. The script overrides data.root/file_pattern in the config.
TEST_H5_PATH = Path(os.environ.get("RAE_TEST_H5", PROJECT_DIR / "data" / "sample.h5"))

OUT_DIR_ROOT = PROJECT_DIR / "outputs"
# Large number means "run all available clips". For this city_0000_25fps.h5
# with clip_len=7 and CLIP_STRIDE=6, that is about 16 clips.
NUM_SAMPLES = 999999
SAMPLING_STEPS = 50
USE_EMA_WEIGHTS = True

# Match training crop by default. The checkpoint was trained with 256x256
# random crops, so 256x256 is the safest test setting.
CROP_SIZE = [256, 256]

# "center" uses the middle 256x256 crop.
# "manual" uses MANUAL_CROP_TOP / MANUAL_CROP_LEFT below.
# "full" disables cropping; this is much heavier and outside the training crop
# distribution, so use it only for quick experiments on a strong GPU.
CROP_MODE = "center"
MANUAL_CROP_TOP = 202
MANUAL_CROP_LEFT = 432

# For a continuous GIF, use stride 6 with clip_len 7. Each clip then predicts
# frames [0..6], [6..12], [12..18], ... and shared endpoint frames are merged.
# Use stride 1 only when you want many overlapping diagnostic clips.
CLIP_STRIDE = 6
SAVE_PER_CLIP_OUTPUTS = True
SAVE_STITCHED_GIF = True

# Keyframes supplied as ground truth inside each 7-frame clip:
# "endpoints", stride 6 -> frames [0, 6], predicts [1, 2, 3, 4, 5]
# "stride",    stride 3 -> frames [0, 3, 6], predicts [1, 2, 4, 5]
# "stride",    stride 2 -> frames [0, 2, 4, 6], predicts [1, 3, 5]
KEYFRAME_POLICY = "endpoints"
KEYFRAME_STRIDE = 6

# Inference-only knobs:
# 1) START_FROM_LINEAR_INTERPOLATION uses endpoint/keyframe linear interpolation
#    as a low-frequency initialization, then adds diffusion noise at
#    START_TIMESTEP_RATIO and denoises. This often stabilizes interpolation
#    tasks, but it makes sampling more conservative.
# 2) LINEAR_BLEND_ALPHA blends the final prediction with plain linear
#    interpolation. This can make demos look better, but it is a postprocess
#    baseline mix, not a clean measurement of the model.
START_FROM_LINEAR_INTERPOLATION = True
START_TIMESTEP_RATIO = 0.5
LINEAR_BLEND_ALPHA = 0.0


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


def save_clip(video: torch.Tensor, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / prefix
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = (video.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    gif_frames = []
    for i, frame in enumerate(frames):
        img = frame[0] if frame.ndim == 3 else frame
        imageio.imwrite(frame_dir / f"{i:03d}.png", img)
        gif_frames.append(img)
    imageio.mimsave(out_dir / f"{prefix}.gif", gif_frames, duration=0.08)


def save_gif(frames: list[np.ndarray], path: Path, duration: float = 0.08) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=duration)


def frames_to_uint8(video: torch.Tensor) -> list[np.ndarray]:
    frames = (video.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    out = []
    for frame in frames:
        out.append(frame[0] if frame.ndim == 3 else frame)
    return out


def batch_start_index(batch: dict[str, object]) -> int:
    value = batch.get("start_index", 0)
    if torch.is_tensor(value):
        return int(value.flatten()[0].detach().cpu().item())
    return int(value)


def add_frames_to_timeline(
    timeline: dict[int, list[np.ndarray]],
    start_index: int,
    frames: list[np.ndarray],
) -> None:
    for local_idx, frame in enumerate(frames):
        timeline.setdefault(start_index + local_idx, []).append(frame.astype(np.float32))


def timeline_to_frames(timeline: dict[int, list[np.ndarray]]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for idx in sorted(timeline):
        stacked = np.stack(timeline[idx], axis=0)
        frames.append(np.clip(stacked.mean(axis=0), 0, 255).astype(np.uint8))
    return frames


def linear_keyframe_video(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Piecewise-linear interpolation between available keyframes.

    Returns [B,T,C,H,W] in intensity range [0,1].
    """
    keyframes = batch["keyframes"].float()
    masks = batch["keyframe_mask"].float()
    out = torch.zeros_like(keyframes)
    bsz, frames_t = keyframes.shape[:2]
    for b in range(bsz):
        frame_is_key = masks[b].flatten(1).amax(dim=1) > 0.5
        key_idx = torch.nonzero(frame_is_key, as_tuple=False).flatten().tolist()
        if not key_idx:
            continue
        if len(key_idx) == 1:
            out[b] = keyframes[b, key_idx[0]]
            continue
        first, last = key_idx[0], key_idx[-1]
        out[b, :first] = keyframes[b, first]
        out[b, last + 1 :] = keyframes[b, last]
        for left, right in zip(key_idx[:-1], key_idx[1:]):
            denom = max(right - left, 1)
            for t in range(left, right + 1):
                w = float(t - left) / float(denom)
                out[b, t] = (1.0 - w) * keyframes[b, left] + w * keyframes[b, right]
    return out


def bcthw_to_btchw(video: torch.Tensor) -> torch.Tensor:
    return video.permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def sample_with_linear_start(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    sampling_steps: int,
    start_timestep_ratio: float,
) -> torch.Tensor:
    from raevdm.models.raevdm import diffusion_to_intensity, intensity_to_diffusion, video_to_bcthw

    frames = video_to_bcthw(batch["frames"].float())
    b, c, frames_t, h, w = frames.shape
    device = frames.device
    key_mask = video_to_bcthw(batch["keyframe_mask"].float())
    keyframes = intensity_to_diffusion(video_to_bcthw(batch["keyframes"].float())) * key_mask
    event_out = model.encode_condition(batch, frames_t)
    extra_condition = model.linear_condition(batch) if hasattr(model, "linear_condition") else None

    linear = intensity_to_diffusion(video_to_bcthw(linear_keyframe_video(batch)))
    start_t = int(round((model.schedule.timesteps - 1) * float(start_timestep_ratio)))
    start_t = min(max(start_t, 1), model.schedule.timesteps - 1)
    timesteps = torch.linspace(start_t, 0, int(sampling_steps), device=device).round().long()
    timesteps = torch.unique_consecutive(timesteps)

    t0 = torch.full((b,), int(timesteps[0].item()), device=device, dtype=torch.long)
    x = model.schedule.q_sample(linear, t0, torch.randn((b, c, frames_t, h, w), device=device))
    x = x * (1.0 - key_mask) + keyframes * key_mask

    for i, t_scalar in enumerate(timesteps):
        t = torch.full((b,), int(t_scalar.item()), device=device, dtype=torch.long)
        eps = model.denoiser(x, t, event_out["features"], keyframes, key_mask, extra_condition=extra_condition)
        x0 = model.schedule.predict_x0_from_eps(x, t, eps).clamp(-1.0, 1.0)
        x0 = x0 * (1.0 - key_mask) + keyframes * key_mask
        if i == len(timesteps) - 1:
            x = x0
        else:
            t_prev_scalar = timesteps[i + 1]
            alpha_prev = model.schedule.alphas_cumprod[int(t_prev_scalar.item())].to(device)
            x = torch.sqrt(alpha_prev) * x0 + torch.sqrt(1.0 - alpha_prev) * eps
            x = x * (1.0 - key_mask) + keyframes * key_mask
    return diffusion_to_intensity(x).permute(0, 2, 1, 3, 4).contiguous()


def residual_params_from_config(config: dict[str, object]) -> dict[str, float]:
    params = dict(config.get("residual_correction", {}) or {})
    return {
        "residual_scale": float(params.get("residual_scale", 0.30)),
        "residual_timestep": float(params.get("residual_timestep", 0)),
        "residual_tanh_temperature": float(params.get("residual_tanh_temperature", 2.0)),
        "event_gate_floor": float(params.get("event_gate_floor", 0.08)),
        "event_gate_power": float(params.get("event_gate_power", 0.50)),
    }


def is_residual_checkpoint(config: dict[str, object]) -> bool:
    finetune = config.get("finetune", {})
    return isinstance(finetune, dict) and finetune.get("mode") == "residual_correction"


def is_flow_fusion_checkpoint(config: dict[str, object]) -> bool:
    finetune = config.get("finetune", {})
    return isinstance(finetune, dict) and finetune.get("mode") == "flow_fusion"


def is_occlusion_refinement_checkpoint(config: dict[str, object]) -> bool:
    finetune = config.get("finetune", {})
    return isinstance(finetune, dict) and finetune.get("mode") == "occlusion_refinement"


def event_gate_bcthw(batch: dict[str, torch.Tensor], frames_t: int, params: dict[str, float]) -> torch.Tensor:
    import torch.nn.functional as F
    from raevdm.models.reliability import collapse_event_steps

    device = batch["frames"].device
    h, w = batch["frames"].shape[-2:]

    if "event_interval_observability" in batch:
        obs = batch["event_interval_observability"].float().permute(0, 2, 1, 3, 4).contiguous().clamp(0.0, 1.0)
        motion = (1.0 - obs).clamp(0.0, 1.0)
        if "event_interval_precision" in batch:
            precision = batch["event_interval_precision"].float().permute(0, 2, 1, 3, 4).contiguous().clamp(0.0, 1.0)
            motion = motion * (0.25 + 0.75 * precision)
    else:
        raw = batch["event_raw"].float().abs().sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        event_bins = batch.get("event_bins", 5)
        if torch.is_tensor(event_bins):
            event_bins = int(event_bins.flatten()[0].detach().cpu().item())
        motion = collapse_event_steps(raw, clip_len=frames_t, event_bins=int(event_bins), reduce="mean").clamp(0.0, 1.0)

    gate = torch.zeros((motion.shape[0], 1, frames_t, h, w), device=device, dtype=motion.dtype)
    intervals = motion.shape[2]
    for frame_idx in range(frames_t):
        if frame_idx == 0:
            value = motion[:, :, 0]
        elif frame_idx >= frames_t - 1:
            value = motion[:, :, -1]
        else:
            left = min(frame_idx - 1, intervals - 1)
            right = min(frame_idx, intervals - 1)
            value = 0.5 * (motion[:, :, left] + motion[:, :, right])
        gate[:, :, frame_idx] = value

    gate = F.avg_pool3d(gate, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
    gate = gate.clamp(0.0, 1.0).pow(float(params["event_gate_power"]))
    gate = (float(params["event_gate_floor"]) + (1.0 - float(params["event_gate_floor"])) * gate).clamp(0.0, 1.0)
    key_mask = batch["keyframe_mask"].float().permute(0, 2, 1, 3, 4).contiguous().clamp(0.0, 1.0)
    return gate * (1.0 - key_mask)


@torch.no_grad()
def sample_residual_correction(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    params: dict[str, float],
) -> torch.Tensor:
    from raevdm.models.raevdm import intensity_to_diffusion, video_to_bcthw

    gt = video_to_bcthw(batch["frames"].float())
    linear = video_to_bcthw(linear_keyframe_video(batch)).clamp(0.0, 1.0)
    key_mask = video_to_bcthw(batch["keyframe_mask"].float()).clamp(0.0, 1.0)
    keyframes_int = video_to_bcthw(batch["keyframes"].float()).clamp(0.0, 1.0)
    keyframes = intensity_to_diffusion(keyframes_int) * key_mask
    bsz, _, frames_t, _, _ = gt.shape

    x = intensity_to_diffusion(linear)
    t = torch.full((bsz,), int(params["residual_timestep"]), device=gt.device, dtype=torch.long)
    event_out = model.encode_condition(batch, frames_t)
    extra_condition = model.linear_condition(batch) if hasattr(model, "linear_condition") else None
    raw = model.denoiser(x, t, event_out["features"], keyframes, key_mask, extra_condition=extra_condition)
    gate = event_gate_bcthw(batch, frames_t, params)
    correction = float(params["residual_scale"]) * torch.tanh(raw / float(params["residual_tanh_temperature"])) * gate
    pred = (linear + correction).clamp(0.0, 1.0)
    pred = pred * (1.0 - key_mask) + keyframes_int * key_mask
    return bcthw_to_btchw(pred)


@torch.no_grad()
def sample_flow_fusion(
    model: torch.nn.Module,
    raft: torch.nn.Module,
    raft_transforms: object,
    batch: dict[str, torch.Tensor],
    residual_params: dict[str, float],
    flow_config: dict[str, object],
) -> torch.Tensor:
    from raevdm.flow_warp import (
        estimate_bidirectional_flow,
        synthesize_bidirectional_warp_video,
        synthesize_occlusion_aware_warp_video,
    )

    keyframes = batch["keyframes"].float()
    image0 = keyframes[:, 0]
    image1 = keyframes[:, -1]
    use_occlusion = bool(getattr(model, "use_occlusion_refinement", False))
    flow_updates = int(flow_config.get("flow_updates_val", 12))
    flow01, flow10 = estimate_bidirectional_flow(
        raft,
        raft_transforms,
        image0,
        image1,
        flow_updates=flow_updates,
    )
    if use_occlusion:
        flow_inputs = synthesize_occlusion_aware_warp_video(
            image0,
            image1,
            flow01,
            flow10,
            frames_t=batch["frames"].shape[1],
        )
        flow_inputs["flow01"] = flow01
        flow_inputs["flow10"] = flow10
        mode = "occlusion_refinement"
    else:
        warped, confidence, time_map = synthesize_bidirectional_warp_video(
            image0,
            image1,
            flow01,
            flow10,
            frames_t=batch["frames"].shape[1],
        )
        flow_inputs = {
            "warped_video": warped,
            "warp_confidence": confidence,
            "time_map": time_map,
            "flow01": flow01,
            "flow10": flow10,
        }
        mode = "flow_fusion"
    outputs = model(
        batch,
        mode=mode,
        flow_inputs=flow_inputs,
        residual_cfg=residual_params,
        base_raft_peak=float(flow_config.get("base_raft_peak", 0.55)),
        detach_base=False,
    )
    return bcthw_to_btchw(outputs["pred"])


def main() -> None:
    add_src_to_path()

    from raevdm.data import RLEDClipDataset
    from raevdm.models import build_model
    from raevdm.utils import load_config, to_device

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if not TEST_H5_PATH.exists():
        raise FileNotFoundError(f"Test H5 not found: {TEST_H5_PATH}")

    checkpoint_path = resolve_checkpoint()
    config = load_config(CONFIG_PATH)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = ckpt.get("config")
    if isinstance(checkpoint_config, dict):
        if "model" in checkpoint_config:
            config["model"] = checkpoint_config["model"]
        if "diffusion" in checkpoint_config:
            config["diffusion"] = checkpoint_config["diffusion"]
        if "finetune" in checkpoint_config:
            config["finetune"] = checkpoint_config["finetune"]
        if "residual_correction" in checkpoint_config:
            config["residual_correction"] = checkpoint_config["residual_correction"]
        if "flow_fusion" in checkpoint_config:
            config["flow_fusion"] = checkpoint_config["flow_fusion"]
        if "occlusion_refinement" in checkpoint_config:
            config["occlusion_refinement"] = checkpoint_config["occlusion_refinement"]

    data_cfg = dict(config["data"])
    data_cfg["root"] = str(TEST_H5_PATH.parent)
    data_cfg["file_pattern"] = TEST_H5_PATH.name
    data_cfg["split"] = "all"
    data_cfg["max_files"] = 1
    data_cfg["random_crop"] = False
    data_cfg["random_flip"] = False
    data_cfg["clip_stride"] = CLIP_STRIDE
    data_cfg["crop_size"] = None if CROP_MODE == "full" else CROP_SIZE
    data_cfg["keyframe_policy"] = KEYFRAME_POLICY
    data_cfg["keyframe_stride"] = KEYFRAME_STRIDE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Project: {PROJECT_DIR}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch/step: {ckpt.get('epoch')}/{ckpt.get('step')}")
    print(f"Use linear condition: {config.get('model', {}).get('use_linear_condition', False)}")
    residual_mode = is_residual_checkpoint(config)
    flow_fusion_mode = is_flow_fusion_checkpoint(config)
    occlusion_mode = is_occlusion_refinement_checkpoint(config)
    residual_params = residual_params_from_config(config)
    inference_mode = (
        "occlusion_refinement"
        if occlusion_mode
        else ("flow_fusion" if flow_fusion_mode else ("residual_correction" if residual_mode else "diffusion"))
    )
    out_dir = OUT_DIR_ROOT / f"FINAL_GT_VS_PRED_{checkpoint_path.stem}"
    print(f"Inference mode: {inference_mode}")
    if residual_mode or flow_fusion_mode or occlusion_mode:
        print(f"Residual params: {residual_params}")
    if flow_fusion_mode or occlusion_mode:
        print(f"Flow fusion params: {config.get('flow_fusion', {})}")
    if occlusion_mode:
        print(f"Occlusion refinement params: {config.get('occlusion_refinement', {})}")
    print(f"Test H5: {TEST_H5_PATH}")
    print(f"Output: {out_dir}")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("Warning: CUDA is not available. CPU inference will be very slow.")

    dataset = RLEDClipDataset(**data_cfg)
    if CROP_MODE == "manual":
        crop_h, crop_w = map(int, CROP_SIZE)

        def fixed_crop(height: int, width: int) -> tuple[int, int, int, int]:
            top = min(max(int(MANUAL_CROP_TOP), 0), max(height - crop_h, 0))
            left = min(max(int(MANUAL_CROP_LEFT), 0), max(width - crop_w, 0))
            return top, left, min(crop_h, height), min(crop_w, width)

        dataset._sample_crop = fixed_crop  # type: ignore[method-assign]

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(config).to(device)
    state_key = "model_ema" if USE_EMA_WEIGHTS and "model_ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()
    raft = None
    raft_transforms = None
    if flow_fusion_mode or occlusion_mode:
        from raevdm.flow_warp import load_frozen_raft_small

        raft, raft_transforms = load_frozen_raft_small(device, RAFT_WEIGHTS_PATH)
    print(f"Loaded weights: {state_key}")
    print(f"Available clips: {len(dataset)}")
    print(f"Crop mode: {CROP_MODE}, crop size: {data_cfg['crop_size']}, clip stride: {CLIP_STRIDE}")
    print(f"Keyframes: policy={KEYFRAME_POLICY}, stride={KEYFRAME_STRIDE}")
    print(f"Linear-start sampling: {START_FROM_LINEAR_INTERPOLATION}, ratio: {START_TIMESTEP_RATIO}")
    print(f"Final linear blend alpha: {LINEAR_BLEND_ALPHA}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stitched_pred: dict[int, list[np.ndarray]] = {}
    stitched_gt: dict[int, list[np.ndarray]] = {}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="infer", dynamic_ncols=True)):
            if i >= NUM_SAMPLES:
                break
            batch = to_device(batch, device)
            if flow_fusion_mode or occlusion_mode:
                pred = sample_flow_fusion(
                    model,
                    raft,
                    raft_transforms,
                    batch,
                    residual_params,
                    dict(config.get("flow_fusion", {}) or {}),
                )
            elif residual_mode:
                pred = sample_residual_correction(model, batch, residual_params)
            elif START_FROM_LINEAR_INTERPOLATION:
                pred = sample_with_linear_start(
                    model,
                    batch,
                    sampling_steps=SAMPLING_STEPS,
                    start_timestep_ratio=START_TIMESTEP_RATIO,
                )
            else:
                pred = model.sample(batch, sampling_steps=SAMPLING_STEPS)
            if LINEAR_BLEND_ALPHA > 0:
                linear = linear_keyframe_video(batch).clamp(0.0, 1.0)
                alpha = min(max(float(LINEAR_BLEND_ALPHA), 0.0), 1.0)
                pred = ((1.0 - alpha) * pred + alpha * linear).clamp(0.0, 1.0)
            gt = batch["frames"]

            pred_frames = frames_to_uint8(pred[0])
            gt_frames = frames_to_uint8(gt[0])

            if SAVE_STITCHED_GIF:
                start_index = batch_start_index(batch)
                add_frames_to_timeline(stitched_pred, start_index, pred_frames)
                add_frames_to_timeline(stitched_gt, start_index, gt_frames)

            if SAVE_PER_CLIP_OUTPUTS:
                sample_dir = out_dir / f"sample_{i:03d}"
                save_clip(gt[0], sample_dir, "ground_truth")
                save_clip(pred[0], sample_dir, "prediction")

    if SAVE_STITCHED_GIF and stitched_pred:
        save_gif(
            timeline_to_frames(stitched_gt),
            out_dir / "ground_truth_full.gif",
        )
        save_gif(
            timeline_to_frames(stitched_pred),
            out_dir / "prediction_full.gif",
        )

    print("Done.")
    print(f"Saved samples to: {out_dir}")
    print(f"Ground truth GIF: {out_dir / 'ground_truth_full.gif'}")
    print(f"Prediction GIF: {out_dir / 'prediction_full.gif'}")


if __name__ == "__main__":
    main()
