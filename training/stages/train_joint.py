from __future__ import annotations

import copy
import hashlib
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


# ==================================================
# Memory-safe final joint continuation from occlusion refinement
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "configs" / "stage_base.yaml"
DATA_ROOT = os.environ.get("RAE_DATA_ROOT", str(PROJECT_DIR / "data" / "RLED"))

RESUME_CHECKPOINT = Path(os.environ.get("RAE_RESUME_CHECKPOINT", PROJECT_DIR / "best_occlusion_refinement.pt"))
RAFT_WEIGHTS_PATH = Path(
    os.environ.get("RAE_RAFT_WEIGHTS", PROJECT_DIR / "raft_small_C_T_V2-01064c6d.pth")
)
RAFT_EXPECTED_SHA256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"

GPU_IDS = os.environ.get("RAE_GPU_IDS", "0,1")
USE_DDP = True
MASTER_PORT = "29661"

RUN_NAME_PREFIX = os.environ.get("RAE_RUN_PREFIX", "raevdm_rled_joint_batch1_from_occlusion_best")
OUTPUT_DIR = Path(os.environ.get("RAE_OUTPUT_DIR", PROJECT_DIR / "outputs"))

# Continue the interrupted joint stage only. Source epoch 011 becomes 012-014.
EPOCHS = 3
REFINER_ONLY_EPOCHS = 0
JOINT_START_EPOCH = 0
MIN_EPOCHS_BEFORE_STOP = EPOCHS
EARLY_STOP_PATIENCE = EPOCHS

# These match the decayed learning-rate scale at the end of epoch 011.
REFINER_LR = 1.5e-5
FLOW_FUSION_LR = 1.5e-6
BASE_LR = 1e-7
MIN_LR_RATIO = 0.50
WARMUP_STEPS = 100
WEIGHT_DECAY = 1e-6
EMA_DECAY = 0.9995

BATCH_SIZE = 1  # per GPU; required for the unfrozen joint stage
GRAD_ACCUM_STEPS = 2  # 1 x 2 GPUs x 2 accumulation = effective batch 4
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
MIXED_PRECISION = True
GRAD_CLIP = 1.0
SAVE_EVERY = 1
LOG_EVERY = 50

CROP_SIZE = [256, 256]
CLIP_LEN = 7
CLIP_STRIDE = 1
EVENT_BINS = 5
KEYFRAME_POLICY = "endpoints"
KEYFRAME_STRIDE = 6

FLOW_UPDATES_TRAIN = 12
FLOW_UPDATES_VAL = 12
BASE_RAFT_PEAK = 0.55

RESIDUAL_CFG = {
    "residual_scale": 0.30,
    "residual_timestep": 0,
    "residual_tanh_temperature": 2.0,
    "event_gate_floor": 0.08,
    "event_gate_power": 0.50,
}

LOSS_WEIGHTS = {
    "recon": 4.0,
    "focus_recon": 4.0,
    "ssim": 0.50,
    "gradient": 0.40,
    "laplacian": 0.15,
    "temporal": 0.25,
    "temporal_accel": 0.10,
    "base_guard": 6.0,
    "best_branch_guard": 2.0,
    "hard_pixels": 1.0,
    "static_correction": 0.10,
    "branch_delta_l1": 0.02,
}
CHARBONNIER_EPS = 1e-3
BASE_GUARD_MARGIN = 0.0015
HARD_PIXEL_RATIO = 0.15
FOCUS_WEIGHT = 2.5

VAL_BATCH_SIZE = 1
VAL_MAX_BATCHES = 128


def add_src_to_path() -> None:
    src = PROJECT_DIR / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def set_cuda_visible_devices() -> None:
    if GPU_IDS:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS


def unwrap_model(model):
    from torch.nn.parallel import DistributedDataParallel as DDP

    return model.module if isinstance(model, DDP) else model


def set_requires_grad(module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def safe_barrier(distributed: bool, local_rank: int) -> None:
    if not distributed:
        return
    import torch.distributed as dist

    try:
        dist.barrier(device_ids=[local_rank])
    except TypeError:
        dist.barrier()


def bcthw_to_btchw(video):
    return video.permute(0, 2, 1, 3, 4).contiguous()


def charbonnier_tensor(x, eps: float):
    import torch

    return torch.sqrt(x * x + eps * eps)


def masked_mean(value, mask, eps: float = 1e-6):
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def spatial_gradient_loss(pred, gt, mask, eps: float):
    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_gt = gt[..., :, 1:] - gt[..., :, :-1]
    dx_mask = mask[..., :, 1:] * mask[..., :, :-1]
    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_gt = gt[..., 1:, :] - gt[..., :-1, :]
    dy_mask = mask[..., 1:, :] * mask[..., :-1, :]
    return 0.5 * (
        masked_mean(charbonnier_tensor(dx_pred - dx_gt, eps), dx_mask)
        + masked_mean(charbonnier_tensor(dy_pred - dy_gt, eps), dy_mask)
    )


def laplacian_loss(pred, gt, mask, eps: float):
    import torch.nn.functional as F

    kernel = pred.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 1, 3, 3)
    pred_lap = F.conv3d(pred, kernel, padding=(0, 1, 1))
    gt_lap = F.conv3d(gt, kernel, padding=(0, 1, 1))
    return masked_mean(charbonnier_tensor(pred_lap - gt_lap, eps), mask)


def temporal_delta_loss(pred, gt, mask, eps: float):
    pred_delta = pred[:, :, 1:] - pred[:, :, :-1]
    gt_delta = gt[:, :, 1:] - gt[:, :, :-1]
    delta_mask = mask[:, :, 1:] * mask[:, :, :-1]
    return masked_mean(charbonnier_tensor(pred_delta - gt_delta, eps), delta_mask)


def temporal_acceleration_loss(pred, gt, mask, eps: float):
    pred_accel = pred[:, :, 2:] - 2.0 * pred[:, :, 1:-1] + pred[:, :, :-2]
    gt_accel = gt[:, :, 2:] - 2.0 * gt[:, :, 1:-1] + gt[:, :, :-2]
    accel_mask = mask[:, :, 2:] * mask[:, :, 1:-1] * mask[:, :, :-2]
    return masked_mean(charbonnier_tensor(pred_accel - gt_accel, eps), accel_mask)


def ssim_loss(pred, gt, mask):
    import torch.nn.functional as F

    mu_x = F.avg_pool3d(pred, (1, 3, 3), stride=1, padding=(0, 1, 1))
    mu_y = F.avg_pool3d(gt, (1, 3, 3), stride=1, padding=(0, 1, 1))
    sigma_x = F.avg_pool3d(pred * pred, (1, 3, 3), stride=1, padding=(0, 1, 1)) - mu_x * mu_x
    sigma_y = F.avg_pool3d(gt * gt, (1, 3, 3), stride=1, padding=(0, 1, 1)) - mu_y * mu_y
    sigma_xy = F.avg_pool3d(pred * gt, (1, 3, 3), stride=1, padding=(0, 1, 1)) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2.0 * mu_x * mu_y + c1)
        * (2.0 * sigma_xy + c2)
        / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)).clamp_min(1e-6)
    )
    return masked_mean(((1.0 - score) * 0.5).clamp(0.0, 1.0), mask)


def hard_pixel_loss(pred, gt, mask, eps: float, ratio: float):
    error = charbonnier_tensor(pred - gt, eps)
    valid = error[mask.expand_as(error) > 0.5]
    if valid.numel() == 0:
        return error.new_tensor(0.0)
    count = max(int(valid.numel() * float(ratio)), 1)
    return valid.topk(count, largest=True, sorted=False).values.mean()


def compute_refinement_losses(outputs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    pred = outputs["pred"]
    gt = outputs["gt"]
    base = outputs["flow_fusion_pred"].detach()
    residual = outputs["residual_pred"].detach()
    warped = outputs["warped_video"].detach()
    mask = (1.0 - outputs["keyframe_mask"]).clamp(0.0, 1.0)
    focus = outputs["refinement_focus"].detach().clamp(0.0, 1.0) * mask
    focus_mask = mask * (1.0 + FOCUS_WEIGHT * focus)

    recon = masked_mean(charbonnier_tensor(pred - gt, CHARBONNIER_EPS), mask)
    focus_recon = masked_mean(
        charbonnier_tensor(pred - gt, CHARBONNIER_EPS),
        focus_mask,
    )
    structural = ssim_loss(pred, gt, focus_mask)
    gradient = spatial_gradient_loss(pred, gt, focus_mask, CHARBONNIER_EPS)
    laplacian = laplacian_loss(pred, gt, focus_mask, CHARBONNIER_EPS)
    temporal = temporal_delta_loss(pred, gt, mask, CHARBONNIER_EPS)
    temporal_accel = temporal_acceleration_loss(pred, gt, mask, CHARBONNIER_EPS)

    pred_err = (pred - gt).abs()
    base_err = (base - gt).abs()
    residual_err = (residual - gt).abs()
    warped_err = (warped - gt).abs()
    base_guard = masked_mean(
        charbonnier_tensor(
            F.relu(pred_err - base_err - BASE_GUARD_MARGIN),
            CHARBONNIER_EPS,
        ),
        mask,
    )
    best_branch_err = torch.minimum(base_err, torch.minimum(residual_err, warped_err))
    best_branch_guard = masked_mean(
        charbonnier_tensor(
            F.relu(pred_err - best_branch_err - BASE_GUARD_MARGIN),
            CHARBONNIER_EPS,
        ),
        focus_mask,
    )
    hard_pixels = hard_pixel_loss(pred, gt, mask, CHARBONNIER_EPS, HARD_PIXEL_RATIO)

    total_correction = outputs["branch_adjustment"] + outputs["refinement_v2"]
    static_mask = mask * (1.0 - focus).clamp(0.0, 1.0)
    static_correction = masked_mean(
        charbonnier_tensor(total_correction, CHARBONNIER_EPS),
        static_mask,
    )
    branch_delta_l1 = masked_mean(
        charbonnier_tensor(outputs["branch_deltas"], CHARBONNIER_EPS),
        mask.expand(-1, outputs["branch_deltas"].shape[1], -1, -1, -1),
    )

    losses = {
        "recon": recon,
        "focus_recon": focus_recon,
        "ssim": structural,
        "gradient": gradient,
        "laplacian": laplacian,
        "temporal": temporal,
        "temporal_accel": temporal_accel,
        "base_guard": base_guard,
        "best_branch_guard": best_branch_guard,
        "hard_pixels": hard_pixels,
        "static_correction": static_correction,
        "branch_delta_l1": branch_delta_l1,
    }
    total = pred.new_tensor(0.0)
    for name, value in losses.items():
        total = total + float(LOSS_WEIGHTS[name]) * value
    losses["total"] = total
    return total, losses


def middle_metrics(pred, gt, key_mask) -> Dict[str, float]:
    import torch

    mask = (1.0 - key_mask.float()).clamp(0.0, 1.0)
    err = (pred.float() - gt.float()) * mask
    denom = mask.sum().clamp_min(1.0)
    mae = err.abs().sum() / denom
    mse = (err * err).sum() / denom
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return {"psnr": float(psnr.cpu()), "mae": float(mae.cpu())}


def focus_mae(pred, gt, key_mask, focus) -> float:
    mask = (1.0 - key_mask.float()).clamp(0.0, 1.0)
    weighted = mask * (0.10 + 0.90 * focus.float().clamp(0.0, 1.0))
    return float(masked_mean((pred.float() - gt.float()).abs(), weighted).cpu())


def temporal_mae(pred, gt, key_mask) -> float:
    # Validation tensors are [B, T, C, H, W], so time is dimension 1.
    mask = (1.0 - key_mask.float()).clamp(0.0, 1.0)
    delta_mask = mask[:, 1:] * mask[:, :-1]
    pred_delta = pred[:, 1:] - pred[:, :-1]
    gt_delta = gt[:, 1:] - gt[:, :-1]
    return float(masked_mean((pred_delta - gt_delta).abs(), delta_mask).cpu())


def resolve_resume_checkpoint() -> Path:
    return Path(RESUME_CHECKPOINT)


def build_config() -> Dict[str, Any]:
    import torch
    from raevdm.utils import load_config

    config = load_config(CONFIG_PATH)
    source = torch.load(resolve_resume_checkpoint(), map_location="cpu", weights_only=False)
    source_config = source.get("config", {})
    for section in (
        "model",
        "diffusion",
        "theory",
        "residual",
        "residual_correction",
        "flow_fusion",
        "occlusion_refinement",
    ):
        if section in source_config:
            config[section] = copy.deepcopy(source_config[section])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_name"] = f"{RUN_NAME_PREFIX}_{timestamp}"
    config["output_dir"] = str(OUTPUT_DIR)

    data = dict(config["data"])
    data.update(
        {
            "root": DATA_ROOT,
            "split": "train",
            "clip_len": CLIP_LEN,
            "clip_stride": CLIP_STRIDE,
            "event_bins": EVENT_BINS,
            "crop_size": CROP_SIZE,
            "random_crop": True,
            "random_flip": True,
            "keyframe_policy": KEYFRAME_POLICY,
            "keyframe_stride": KEYFRAME_STRIDE,
        }
    )
    config["data"] = data

    model_cfg = dict(config.get("model", {}))
    model_cfg.update(
        {
            "use_linear_condition": True,
            "use_flow_fusion": True,
            "use_occlusion_refinement": True,
            "occlusion_refinement_hidden_channels": 32,
            "occlusion_refinement_scale": 0.06,
            "occlusion_branch_delta_scale": 0.30,
        }
    )
    config["model"] = model_cfg

    train = dict(config["train"])
    train.update(
        {
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "epochs": EPOCHS,
            "refiner_lr": REFINER_LR,
            "flow_fusion_lr": FLOW_FUSION_LR,
            "base_lr": BASE_LR,
            "warmup_steps": WARMUP_STEPS,
            "ema_decay": EMA_DECAY,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "mixed_precision": MIXED_PRECISION,
            "save_every": SAVE_EVERY,
            "log_every": LOG_EVERY,
        }
    )
    config["train"] = train
    config["finetune"] = {
        "mode": "occlusion_refinement",
        "resume_checkpoint": str(resolve_resume_checkpoint()),
        "continuation": "joint_batch1_grad_accum2",
        "continuation_epochs": EPOCHS,
        "refiner_only_epochs": REFINER_ONLY_EPOCHS,
        "joint_start_epoch": JOINT_START_EPOCH,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "loss_weights": dict(LOSS_WEIGHTS),
    }
    config["residual_correction"] = dict(RESIDUAL_CFG)
    config["flow_fusion"] = {
        "raft_architecture": "raft_small",
        "raft_weights_file": RAFT_WEIGHTS_PATH.name,
        "flow_updates_train": FLOW_UPDATES_TRAIN,
        "flow_updates_val": FLOW_UPDATES_VAL,
        "base_raft_peak": BASE_RAFT_PEAK,
        "raft_frozen": True,
        "occlusion_aux_version": 2,
    }
    config["occlusion_refinement"] = {
        "architecture": "multiscale_spatiotemporal_v1",
        "zero_initialized": True,
        "focus_weight": FOCUS_WEIGHT,
        "hard_pixel_ratio": HARD_PIXEL_RATIO,
    }
    return config


def build_datasets(config: Dict[str, Any]):
    from raevdm.data import RLEDClipDataset

    train_cfg = dict(config["data"])
    train_cfg["split"] = "train"
    train_set = RLEDClipDataset(**train_cfg)
    val_cfg = dict(config["data"])
    val_cfg.update({"split": "val", "random_crop": False, "random_flip": False})
    try:
        val_set = RLEDClipDataset(**val_cfg)
    except Exception as exc:
        print(f"[warn] could not build val split ({exc}); using all split center crops.")
        val_cfg["split"] = "all"
        val_set = RLEDClipDataset(**val_cfg)
    return train_set, val_set


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    import torch

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-6)
        denom = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def load_occlusion_resume(model, checkpoint_path: Path, device) -> Dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_mode = checkpoint.get("config", {}).get("finetune", {}).get("mode")
    if source_mode != "occlusion_refinement":
        raise RuntimeError(
            "Continuation must start from an occlusion-refinement checkpoint. "
            f"Expected mode='occlusion_refinement', got {source_mode!r}."
        )
    state_key = "model_ema" if "model_ema" in checkpoint else "model"
    source_state = checkpoint[state_key]
    target_state = model.state_dict()
    missing = []
    mismatches = []
    for key, target in target_state.items():
        source = source_state.get(key)
        if source is None:
            missing.append(key)
        elif tuple(source.shape) != tuple(target.shape):
            mismatches.append((key, tuple(source.shape), tuple(target.shape)))
    unexpected_source = [key for key in source_state if key not in target_state]
    if missing or mismatches:
        raise RuntimeError(
            f"Incomplete continuation checkpoint: missing={missing[:8]}, "
            f"shape_mismatches={mismatches[:8]}"
        )
    if unexpected_source:
        raise RuntimeError(f"Unexpected source tensors: {unexpected_source[:16]}")
    model.load_state_dict(source_state, strict=True)
    model.to(device)
    print(
        f"Loaded occlusion-refinement continuation from {checkpoint_path} ({state_key}), "
        f"epoch={checkpoint.get('epoch')}, step={checkpoint.get('step')}, "
        f"loaded={len(target_state)} tensors"
    )
    return checkpoint


def build_flow_inputs(raft, transforms, batch: Dict[str, Any], flow_updates: int) -> Dict[str, Any]:
    from raevdm.flow_warp import (
        estimate_bidirectional_flow,
        synthesize_occlusion_aware_warp_video,
    )

    keyframes = batch["keyframes"].float()
    image0 = keyframes[:, 0]
    image1 = keyframes[:, -1]
    flow01, flow10 = estimate_bidirectional_flow(
        raft,
        transforms,
        image0,
        image1,
        flow_updates=flow_updates,
    )
    flow_inputs = synthesize_occlusion_aware_warp_video(
        image0,
        image1,
        flow01,
        flow10,
        frames_t=batch["frames"].shape[1],
    )
    flow_inputs["flow01"] = flow01
    flow_inputs["flow10"] = flow10
    return {key: value.detach() for key, value in flow_inputs.items()}


def forward_model(model, batch, flow_inputs, detach_base: bool):
    return model(
        batch,
        mode="occlusion_refinement",
        flow_inputs=flow_inputs,
        residual_cfg=RESIDUAL_CFG,
        base_raft_peak=BASE_RAFT_PEAK,
        detach_base=detach_base,
    )


def update_ema(ema_model, model, decay: float) -> None:
    import torch

    with torch.no_grad():
        source_state = unwrap_model(model).state_dict()
        for key, value in ema_model.state_dict().items():
            source = source_state[key].detach()
            if torch.is_floating_point(value):
                value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                value.copy_(source)


def validate(model, raft, transforms, loader, device, max_batches: int) -> Dict[str, float]:
    import torch
    from raevdm.utils import to_device

    was_training = model.training
    model.eval()
    sums = {
        "refined_middle_psnr": 0.0,
        "refined_middle_mae": 0.0,
        "base_fusion_middle_psnr": 0.0,
        "base_fusion_middle_mae": 0.0,
        "residual_middle_psnr": 0.0,
        "residual_middle_mae": 0.0,
        "warp_middle_psnr": 0.0,
        "warp_middle_mae": 0.0,
        "focus_middle_mae": 0.0,
        "temporal_middle_mae": 0.0,
    }
    count = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= int(max_batches):
                break
            batch = to_device(batch, device)
            flow_inputs = build_flow_inputs(raft, transforms, batch, FLOW_UPDATES_VAL)
            outputs = forward_model(model, batch, flow_inputs, detach_base=False)
            gt = bcthw_to_btchw(outputs["gt"])
            key_mask = bcthw_to_btchw(outputs["keyframe_mask"])
            videos = {
                "refined": bcthw_to_btchw(outputs["pred"]),
                "base_fusion": bcthw_to_btchw(outputs["flow_fusion_pred"]),
                "residual": bcthw_to_btchw(outputs["residual_pred"]),
                "warp": bcthw_to_btchw(outputs["warped_video"]),
            }
            for name, video in videos.items():
                result = middle_metrics(video, gt, key_mask)
                sums[f"{name}_middle_psnr"] += result["psnr"]
                sums[f"{name}_middle_mae"] += result["mae"]
            focus = bcthw_to_btchw(outputs["refinement_focus"])
            sums["focus_middle_mae"] += focus_mae(videos["refined"], gt, key_mask, focus)
            sums["temporal_middle_mae"] += temporal_mae(videos["refined"], gt, key_mask)
            count += 1

    if was_training:
        model.train()
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    metrics["psnr_gain_vs_base"] = (
        metrics["refined_middle_psnr"] - metrics["base_fusion_middle_psnr"]
    )
    metrics["mae_gain_vs_base"] = (
        metrics["base_fusion_middle_mae"] - metrics["refined_middle_mae"]
    )
    return metrics


def save_payload(path, model, optimizer, epoch, step, config, extra) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
    }
    payload.update(extra)
    torch.save(payload, path)


def save_inference_payload(path, ema_model, epoch, step, config, metrics) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_ema": ema_model.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
            "finetune_metrics": metrics,
            "inference_only": True,
        },
        path,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(config: Dict[str, Any]) -> None:
    import torch

    checkpoint_path = resolve_resume_checkpoint()
    for path, label in (
        (CONFIG_PATH, "Config"),
        (checkpoint_path, "Occlusion-refinement checkpoint"),
        (RAFT_WEIGHTS_PATH, "RAFT weights"),
        (Path(DATA_ROOT), "Training data root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    raft_hash = file_sha256(RAFT_WEIGHTS_PATH)
    if raft_hash.lower() != RAFT_EXPECTED_SHA256.lower():
        raise RuntimeError(
            "RAFT weight checksum mismatch. "
            f"Expected {RAFT_EXPECTED_SHA256}, got {raft_hash}."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mode = checkpoint.get("config", {}).get("finetune", {}).get("mode")
    if mode != "occlusion_refinement":
        raise RuntimeError(
            f"Checkpoint mode is {mode!r}, expected 'occlusion_refinement'."
        )
    if "model_ema" not in checkpoint:
        raise RuntimeError("Checkpoint has no model_ema weights.")
    try:
        from torchvision.models.optical_flow import raft_small  # noqa: F401
    except Exception as exc:
        raise RuntimeError("torchvision RAFT is unavailable in RaE_env.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    requested = [item for item in GPU_IDS.split(",") if item.strip()]
    if USE_DDP and torch.cuda.device_count() < len(requested):
        raise RuntimeError(
            f"Requested {len(requested)} GPUs, only {torch.cuda.device_count()} visible."
        )
    print("Preflight passed.")
    print(f"Occlusion-refinement checkpoint: {checkpoint_path}")
    print(f"Source epoch/step: {checkpoint.get('epoch')}/{checkpoint.get('step')}")
    print(f"RAFT weights: {RAFT_WEIGHTS_PATH} (sha256 OK)")
    print(f"Run name: {config['run_name']}")


def configure_stage(model, epoch: int) -> Tuple[str, bool]:
    raw = unwrap_model(model)
    set_requires_grad(raw, False)
    set_requires_grad(raw.occlusion_refinement, True)
    if epoch < REFINER_ONLY_EPOCHS:
        return "refiner", True
    set_requires_grad(raw.flow_fusion, True)
    if epoch < JOINT_START_EPOCH:
        return "refiner+fusion", True
    set_requires_grad(raw.event_encoder, True)
    set_requires_grad(raw.denoiser, True)
    return "joint", False


def worker(local_rank: int, world_size: int, config: Dict[str, Any]) -> None:
    add_src_to_path()
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from tqdm import tqdm

    from raevdm.flow_warp import load_frozen_raft_small
    from raevdm.models import build_model
    from raevdm.utils import count_parameters, save_config, set_seed, to_device

    distributed = world_size > 1
    if distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", MASTER_PORT)
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda", 0)

    is_main = local_rank == 0
    set_seed(int(config.get("seed", 3407)) + local_rank)
    run_dir = Path(config["output_dir"]) / config["run_name"]
    if is_main:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
        save_config(config, run_dir / "config_occlusion_refinement.yaml")
        print(f"Run dir: {run_dir}")
        print(f"World size: {world_size}")
        print(f"Data root: {DATA_ROOT}")
    safe_barrier(distributed, local_rank)

    train_set, val_set = build_datasets(config)
    sampler = (
        DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=True,
        )
        if distributed
        else None
    )
    train_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "drop_last": True,
    }
    if NUM_WORKERS > 0:
        train_kwargs.update(
            {"persistent_workers": True, "prefetch_factor": PREFETCH_FACTOR}
        )
    train_loader = DataLoader(train_set, **train_kwargs)
    val_loader = DataLoader(
        val_set,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = build_model(config)
    source_checkpoint = load_occlusion_resume(
        model,
        resolve_resume_checkpoint(),
        device,
    )
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    raft, raft_transforms = load_frozen_raft_small(device, RAFT_WEIGHTS_PATH)

    raw_model = unwrap_model(model)
    refiner_parameters = list(raw_model.occlusion_refinement.parameters())
    refiner_ids = {id(parameter) for parameter in refiner_parameters}
    fusion_parameters = list(raw_model.flow_fusion.parameters())
    fusion_ids = {id(parameter) for parameter in fusion_parameters}
    base_parameters = [
        parameter
        for parameter in raw_model.parameters()
        if id(parameter) not in refiner_ids and id(parameter) not in fusion_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": refiner_parameters, "lr": REFINER_LR, "name": "refiner"},
            {"params": fusion_parameters, "lr": FLOW_FUSION_LR, "name": "flow_fusion"},
            {"params": base_parameters, "lr": BASE_LR, "name": "base"},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_steps_per_epoch = max(
        math.ceil(len(train_loader) / GRAD_ACCUM_STEPS),
        1,
    )
    total_steps = optimizer_steps_per_epoch * EPOCHS
    scheduler = make_scheduler(optimizer, total_steps, WARMUP_STEPS)
    scaler = torch.amp.GradScaler("cuda", enabled=MIXED_PRECISION)
    ema_model = copy.deepcopy(raw_model).to(device).eval()
    set_requires_grad(ema_model, False)

    if is_main:
        print(f"Train clips: {len(train_set)}")
        print(f"Val clips: {len(val_set)}")
        print(f"Total parameters: {count_parameters(raw_model) / 1e6:.2f} M")
        print(
            "Refiner parameters: "
            f"{sum(p.numel() for p in refiner_parameters) / 1e6:.3f} M"
        )
        print(f"Micro batch size per GPU: {BATCH_SIZE}")
        print(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
        print(
            "Effective global batch size: "
            f"{BATCH_SIZE * world_size * GRAD_ACCUM_STEPS}"
        )
        print(f"Stage: memory-safe joint fine-tuning for {EPOCHS} epochs")
        print(f"Loss weights: {LOSS_WEIGHTS}")

    source_epoch = int(source_checkpoint.get("epoch", 11))
    first_epoch = source_epoch + 1
    global_step = int(source_checkpoint.get("step", 0))
    best_psnr = -1e9
    best_focus = float("inf")
    best_temporal = float("inf")
    no_improve = 0

    if is_main:
        initial_metrics = validate(
            ema_model,
            raft,
            raft_transforms,
            val_loader,
            device,
            VAL_MAX_BATCHES,
        )
        print(f"initial val={initial_metrics}")
        extra = {
            "model_ema": ema_model.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "finetune_metrics": initial_metrics,
            "flow_fusion": dict(config["flow_fusion"]),
            "occlusion_refinement": dict(config["occlusion_refinement"]),
        }
        checkpoint_dir = run_dir / "checkpoints"
        for filename in (
            "source_baseline.pt",
            "best_middle.pt",
            "best_occlusion_refinement.pt",
            "best_focus.pt",
            "best_temporal.pt",
        ):
            save_payload(
                checkpoint_dir / filename,
                model,
                optimizer,
                source_epoch,
                global_step,
                config,
                extra,
            )
        save_inference_payload(
            checkpoint_dir / "FINAL_BEST_INFERENCE.pt",
            ema_model,
            source_epoch,
            global_step,
            config,
            initial_metrics,
        )
        best_psnr = initial_metrics["refined_middle_psnr"]
        best_focus = initial_metrics["focus_middle_mae"]
        best_temporal = initial_metrics["temporal_middle_mae"]
        print(f"initial best refined middle PSNR: {best_psnr:.4f}")
    safe_barrier(distributed, local_rank)

    for local_epoch in range(EPOCHS):
        epoch = first_epoch + local_epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        stage, detach_base = configure_stage(model, local_epoch)
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        start = time.perf_counter()
        pbar = tqdm(
            train_loader,
            desc=f"occlusion {stage} epoch {epoch:03d}",
            dynamic_ncols=True,
            disable=not is_main,
        )

        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(pbar):
            batch = to_device(batch, device)
            with torch.no_grad():
                flow_inputs = build_flow_inputs(
                    raft,
                    raft_transforms,
                    batch,
                    FLOW_UPDATES_TRAIN,
                )
            group_start = (batch_index // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
            accumulation_size = min(
                GRAD_ACCUM_STEPS,
                len(train_loader) - group_start,
            )
            should_step = (
                (batch_index + 1) % GRAD_ACCUM_STEPS == 0
                or batch_index + 1 == len(train_loader)
            )
            sync_context = (
                nullcontext()
                if should_step or not distributed
                else model.no_sync()
            )
            with sync_context:
                with torch.amp.autocast("cuda", enabled=MIXED_PRECISION):
                    outputs = forward_model(
                        model,
                        batch,
                        flow_inputs,
                        detach_base=detach_base,
                    )
                    total, losses = compute_refinement_losses(outputs)
                    backward_loss = total / accumulation_size
                if not torch.isfinite(total):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch}, step={global_step}: "
                        f"{total}"
                    )
                scaler.scale(backward_loss).backward()

            epoch_loss += float(total.detach().cpu())
            epoch_count += 1
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GRAD_CLIP,
                )
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                    update_ema(ema_model, model, EMA_DECAY)
                global_step += 1
            if is_main and should_step and global_step % LOG_EVERY == 0:
                text = " ".join(
                    f"{key}:{float(value.detach().cpu()):.4f}"
                    for key, value in losses.items()
                )
                pbar.set_postfix_str(text)

        safe_barrier(distributed, local_rank)
        stop_now = False
        if is_main:
            metrics = validate(
                ema_model,
                raft,
                raft_transforms,
                val_loader,
                device,
                VAL_MAX_BATCHES,
            )
            average_loss = epoch_loss / max(epoch_count, 1)
            elapsed = (time.perf_counter() - start) / 60.0
            print(
                f"epoch {epoch:03d} stage={stage} loss={average_loss:.5f} "
                f"elapsed={elapsed:.1f}min val={metrics}"
            )
            extra = {
                "model_ema": ema_model.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "finetune_metrics": metrics,
                "flow_fusion": dict(config["flow_fusion"]),
                "occlusion_refinement": dict(config["occlusion_refinement"]),
            }
            checkpoint_dir = run_dir / "checkpoints"
            save_payload(
                checkpoint_dir / "latest.pt",
                model,
                optimizer,
                epoch,
                global_step,
                config,
                extra,
            )
            if (local_epoch + 1) % SAVE_EVERY == 0:
                save_payload(
                    checkpoint_dir / f"epoch_{epoch:03d}.pt",
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    config,
                    extra,
                )

            score = metrics["refined_middle_psnr"]
            if score > best_psnr + 1e-4:
                best_psnr = score
                no_improve = 0
                for filename in ("best_middle.pt", "best_occlusion_refinement.pt"):
                    save_payload(
                        checkpoint_dir / filename,
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        config,
                        extra,
                    )
                save_inference_payload(
                    checkpoint_dir / "FINAL_BEST_INFERENCE.pt",
                    ema_model,
                    epoch,
                    global_step,
                    config,
                    metrics,
                )
                print(f"new best refined middle PSNR: {best_psnr:.4f}")
            else:
                no_improve += 1
                print(
                    f"no PSNR improvement for {no_improve}/"
                    f"{EARLY_STOP_PATIENCE} epochs"
                )

            if metrics["focus_middle_mae"] < best_focus - 1e-6:
                best_focus = metrics["focus_middle_mae"]
                save_payload(
                    checkpoint_dir / "best_focus.pt",
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    config,
                    extra,
                )
                print(f"new best focus MAE: {best_focus:.6f}")
            if metrics["temporal_middle_mae"] < best_temporal - 1e-6:
                best_temporal = metrics["temporal_middle_mae"]
                save_payload(
                    checkpoint_dir / "best_temporal.pt",
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    config,
                    extra,
                )
                print(f"new best temporal MAE: {best_temporal:.6f}")

            if (
                local_epoch + 1 >= MIN_EPOCHS_BEFORE_STOP
                and no_improve >= EARLY_STOP_PATIENCE
            ):
                stop_now = True
                print("early stopping: refined validation PSNR stopped improving.")

        if distributed:
            flag = torch.tensor(
                [1 if stop_now else 0],
                device=device,
                dtype=torch.int32,
            )
            dist.broadcast(flag, src=0)
            stop_now = bool(flag.item())
        if stop_now:
            break
        safe_barrier(distributed, local_rank)

    if distributed:
        dist.destroy_process_group()


def main() -> None:
    set_cuda_visible_devices()
    add_src_to_path()
    config = build_config()
    preflight(config)

    import torch
    import torch.multiprocessing as mp

    requested = [item.strip() for item in GPU_IDS.split(",") if item.strip()]
    world_size = len(requested) if USE_DDP and len(requested) > 1 else 1
    if world_size > 1:
        mp.spawn(worker, nprocs=world_size, args=(world_size, config), join=True)
    else:
        worker(0, 1, config)


if __name__ == "__main__":
    main()
