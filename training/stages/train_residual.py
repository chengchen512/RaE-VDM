from __future__ import annotations

import copy
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


# ==============================
# Residual-correction fine-tune setup
# ==============================

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "configs" / "stage_base.yaml"
DATA_ROOT = os.environ.get("RAE_DATA_ROOT", str(PROJECT_DIR / "data" / "RLED"))

RESUME_CHECKPOINT = Path(
    os.environ.get("RAE_RESUME_CHECKPOINT", PROJECT_DIR / "core_checkpoint.pt")
)

GPU_IDS = os.environ.get("RAE_GPU_IDS", "0,1")
USE_DDP = True
MASTER_PORT = "29641"

RUN_NAME_PREFIX = os.environ.get("RAE_RUN_PREFIX", "raevdm_rled_residual_from_core")
OUTPUT_DIR = Path(os.environ.get("RAE_OUTPUT_DIR", PROJECT_DIR / "outputs"))

# This is a new objective, so resume weights only. Do not resume the old Adam
# state from diffusion training.
LOAD_EMA_WEIGHTS = True
EPOCHS = 20
EARLY_STOP_PATIENCE = 8
LR = 2e-5
MIN_LR = 2e-6
WARMUP_STEPS = 1000
EMA_DECAY = 0.9995
WEIGHT_DECAY = 1e-6

BATCH_SIZE = 2  # per GPU
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
USE_LINEAR_CONDITION = True

# Residual correction: final = linear_keyframe_interp + bounded residual.
# A bounded residual makes it hard for the model to destroy a strong linear
# baseline, while still allowing event-guided corrections around motion.
RESIDUAL_SCALE = 0.30
RESIDUAL_TIMESTEP = 0
RESIDUAL_TANH_TEMPERATURE = 2.0
EVENT_GATE_FLOOR = 0.08
EVENT_GATE_POWER = 0.50
CHARBONNIER_EPS = 1e-3

LOSS_WEIGHTS = {
    "recon": 5.0,
    "residual": 1.0,
    "linear_guard": 4.0,
    "linear_preserve": 1.5,
    "gradient": 0.20,
    "temporal": 0.20,
    "quiet_residual": 0.40,
    "residual_tv": 0.05,
}
LINEAR_GUARD_MARGIN = 0.003
LINEAR_GOOD_THRESHOLD = 0.035

# Residual validation is cheap, so validate on more clips than the diffusion
# sampler version. This gives a more stable "best_middle.pt" choice.
VAL_EVERY = 1
VAL_BATCH_SIZE = 4
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


def set_requires_grad(model, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def safe_barrier(distributed: bool, local_rank: int) -> None:
    if not distributed:
        return
    import torch.distributed as dist

    try:
        dist.barrier(device_ids=[local_rank])
    except TypeError:
        dist.barrier()


def video_to_bcthw(video):
    return video.permute(0, 2, 1, 3, 4).contiguous()


def bcthw_to_btchw(video):
    return video.permute(0, 2, 1, 3, 4).contiguous()


def charbonnier_tensor(x, eps: float):
    import torch

    return torch.sqrt(x * x + eps * eps)


def masked_mean(value, mask, eps: float = 1e-6):
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def linear_keyframe_video(batch: Dict[str, Any]):
    import torch

    keyframes = batch["keyframes"].float()
    masks = batch["keyframe_mask"].float()
    out = torch.zeros_like(keyframes)
    bsz = keyframes.shape[0]
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


def event_gate_bcthw(batch: Dict[str, Any], frames_t: int):
    import torch
    import torch.nn.functional as F
    from raevdm.models.reliability import collapse_event_steps

    device = batch["frames"].device
    h, w = batch["frames"].shape[-2:]

    if "event_interval_observability" in batch:
        obs = video_to_bcthw(batch["event_interval_observability"].float()).clamp(0.0, 1.0)
        motion = (1.0 - obs).clamp(0.0, 1.0)
        if "event_interval_precision" in batch:
            precision = video_to_bcthw(batch["event_interval_precision"].float()).clamp(0.0, 1.0)
            motion = motion * (0.25 + 0.75 * precision)
    else:
        raw = batch["event_raw"].float().abs().sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        event_bins = int(batch.get("event_bins", EVENT_BINS))
        if torch.is_tensor(batch.get("event_bins", None)):
            event_bins = int(batch["event_bins"].flatten()[0].detach().cpu().item())
        motion = collapse_event_steps(raw, clip_len=frames_t, event_bins=event_bins, reduce="mean").clamp(0.0, 1.0)

    gate = torch.zeros((motion.shape[0], 1, frames_t, h, w), device=device, dtype=motion.dtype)
    intervals = motion.shape[2]
    for frame_idx in range(frames_t):
        if intervals <= 0:
            value = motion.new_zeros((motion.shape[0], 1, h, w))
        elif frame_idx == 0:
            value = motion[:, :, 0]
        elif frame_idx >= frames_t - 1:
            value = motion[:, :, -1]
        else:
            left = min(frame_idx - 1, intervals - 1)
            right = min(frame_idx, intervals - 1)
            value = 0.5 * (motion[:, :, left] + motion[:, :, right])
        gate[:, :, frame_idx] = value

    gate = F.avg_pool3d(gate, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
    gate = gate.clamp(0.0, 1.0).pow(float(EVENT_GATE_POWER))
    gate = (float(EVENT_GATE_FLOOR) + (1.0 - float(EVENT_GATE_FLOOR)) * gate).clamp(0.0, 1.0)
    key_mask = video_to_bcthw(batch["keyframe_mask"].float()).clamp(0.0, 1.0)
    return gate * (1.0 - key_mask)


def residual_forward(model, batch: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from raevdm.models.raevdm import intensity_to_diffusion

    model_core = unwrap_model(model)
    gt = video_to_bcthw(batch["frames"].float())
    linear = video_to_bcthw(linear_keyframe_video(batch)).clamp(0.0, 1.0)
    key_mask = video_to_bcthw(batch["keyframe_mask"].float()).clamp(0.0, 1.0)
    keyframes_int = video_to_bcthw(batch["keyframes"].float()).clamp(0.0, 1.0)
    keyframes = intensity_to_diffusion(keyframes_int) * key_mask
    bsz, channels, frames_t, h, w = gt.shape

    x = intensity_to_diffusion(linear)
    t = torch.full((bsz,), int(RESIDUAL_TIMESTEP), device=gt.device, dtype=torch.long)
    event_out = model_core.encode_condition(batch, frames_t)
    extra_condition = model_core.linear_condition(batch) if hasattr(model_core, "linear_condition") else None
    raw_residual = model_core.denoiser(
        x,
        t,
        event_out["features"],
        keyframes,
        key_mask,
        extra_condition=extra_condition,
    )
    gate = event_gate_bcthw(batch, frames_t)
    correction = float(RESIDUAL_SCALE) * torch.tanh(raw_residual / float(RESIDUAL_TANH_TEMPERATURE)) * gate
    correction = correction * (1.0 - key_mask)
    pred_unclamped = linear + correction
    pred = pred_unclamped.clamp(0.0, 1.0)
    pred = pred * (1.0 - key_mask) + keyframes_int * key_mask
    return {
        "pred": pred,
        "pred_unclamped": pred_unclamped,
        "linear": linear,
        "correction": correction,
        "raw_residual": raw_residual,
        "gate": gate,
        "gt": gt,
        "key_mask": key_mask,
        "event": event_out,
    }


def spatial_gradient_loss(pred, gt, mask, eps: float):
    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_gt = gt[..., :, 1:] - gt[..., :, :-1]
    dx_mask = mask[..., :, 1:] * mask[..., :, :-1]

    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_gt = gt[..., 1:, :] - gt[..., :-1, :]
    dy_mask = mask[..., 1:, :] * mask[..., :-1, :]

    loss_x = masked_mean(charbonnier_tensor(dx_pred - dx_gt, eps), dx_mask)
    loss_y = masked_mean(charbonnier_tensor(dy_pred - dy_gt, eps), dy_mask)
    return 0.5 * (loss_x + loss_y)


def temporal_delta_loss(pred, gt, mask, eps: float):
    if pred.shape[2] < 2:
        return pred.new_tensor(0.0)
    pred_delta = pred[:, :, 1:] - pred[:, :, :-1]
    gt_delta = gt[:, :, 1:] - gt[:, :, :-1]
    delta_mask = mask[:, :, 1:] * mask[:, :, :-1]
    return masked_mean(charbonnier_tensor(pred_delta - gt_delta, eps), delta_mask)


def residual_tv_loss(correction, mask, eps: float):
    dx = correction[..., :, 1:] - correction[..., :, :-1]
    dx_mask = mask[..., :, 1:] * mask[..., :, :-1]
    dy = correction[..., 1:, :] - correction[..., :-1, :]
    dy_mask = mask[..., 1:, :] * mask[..., :-1, :]
    dt = correction[:, :, 1:] - correction[:, :, :-1] if correction.shape[2] > 1 else None
    if dt is None:
        return 0.5 * (
            masked_mean(charbonnier_tensor(dx, eps), dx_mask)
            + masked_mean(charbonnier_tensor(dy, eps), dy_mask)
        )
    dt_mask = mask[:, :, 1:] * mask[:, :, :-1]
    return (
        masked_mean(charbonnier_tensor(dx, eps), dx_mask)
        + masked_mean(charbonnier_tensor(dy, eps), dy_mask)
        + 0.5 * masked_mean(charbonnier_tensor(dt, eps), dt_mask)
    ) / 2.5


def compute_residual_losses(batch: Dict[str, Any], outputs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    import torch.nn.functional as F

    pred = outputs["pred"]
    gt = outputs["gt"]
    linear = outputs["linear"]
    correction = outputs["correction"]
    gate = outputs["gate"]
    mask = (1.0 - outputs["key_mask"]).clamp(0.0, 1.0)

    target_residual = (gt - linear).clamp(-float(RESIDUAL_SCALE), float(RESIDUAL_SCALE)) * mask

    recon = masked_mean(charbonnier_tensor(pred - gt, CHARBONNIER_EPS), mask)
    residual = masked_mean(charbonnier_tensor(correction - target_residual, CHARBONNIER_EPS), mask)

    pred_err = (pred - gt).abs()
    linear_err = (linear - gt).abs().detach()
    excess = F.relu(pred_err - linear_err - float(LINEAR_GUARD_MARGIN))
    linear_guard = masked_mean(charbonnier_tensor(excess, CHARBONNIER_EPS), mask)

    good_linear_mask = ((linear_err < float(LINEAR_GOOD_THRESHOLD)).float() * mask).detach()
    linear_preserve = masked_mean(charbonnier_tensor(correction, CHARBONNIER_EPS), good_linear_mask)

    gradient = spatial_gradient_loss(pred, gt, mask, CHARBONNIER_EPS)
    temporal = temporal_delta_loss(pred, gt, mask, CHARBONNIER_EPS)
    quiet_residual = masked_mean(charbonnier_tensor(correction, CHARBONNIER_EPS), (1.0 - gate).detach() * mask)
    residual_tv = residual_tv_loss(correction, mask, CHARBONNIER_EPS)

    losses = {
        "recon": recon,
        "residual": residual,
        "linear_guard": linear_guard,
        "linear_preserve": linear_preserve,
        "gradient": gradient,
        "temporal": temporal,
        "quiet_residual": quiet_residual,
        "residual_tv": residual_tv,
    }
    total = pred.new_tensor(0.0)
    for name, value in losses.items():
        total = total + float(LOSS_WEIGHTS.get(name, 0.0)) * value
    losses["total"] = total
    return total, losses


def middle_metrics_btchw(pred, gt, key_mask) -> Dict[str, float]:
    import torch

    mask = (1.0 - key_mask.float()).clamp(0.0, 1.0)
    err = (pred.float() - gt.float()) * mask
    denom = mask.sum().clamp_min(1.0)
    mae = err.abs().sum() / denom
    mse = (err * err).sum() / denom
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return {
        "middle_mae": float(mae.detach().cpu()),
        "middle_psnr": float(psnr.detach().cpu()),
    }


def build_config() -> Dict[str, Any]:
    from raevdm.utils import load_config

    config = load_config(CONFIG_PATH)
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
    model_cfg["use_linear_condition"] = bool(USE_LINEAR_CONDITION)
    config["model"] = model_cfg

    train = dict(config["train"])
    train.update(
        {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "epochs": EPOCHS,
            "lr": LR,
            "min_lr": MIN_LR,
            "warmup_steps": WARMUP_STEPS,
            "ema_decay": EMA_DECAY,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "mixed_precision": MIXED_PRECISION,
            "save_every": SAVE_EVERY,
            "log_every": LOG_EVERY,
            "scheduler": "cosine",
        }
    )
    config["train"] = train

    config["finetune"] = {
        "mode": "residual_correction",
        "resume_checkpoint": str(resolve_resume_checkpoint()),
        "load_ema_weights": LOAD_EMA_WEIGHTS,
        "loss_weights": dict(LOSS_WEIGHTS),
        "early_stop_patience": EARLY_STOP_PATIENCE,
    }
    config["residual_correction"] = {
        "residual_scale": RESIDUAL_SCALE,
        "residual_timestep": RESIDUAL_TIMESTEP,
        "residual_tanh_temperature": RESIDUAL_TANH_TEMPERATURE,
        "event_gate_floor": EVENT_GATE_FLOOR,
        "event_gate_power": EVENT_GATE_POWER,
        "linear_guard_margin": LINEAR_GUARD_MARGIN,
        "linear_good_threshold": LINEAR_GOOD_THRESHOLD,
    }
    return config


def resolve_resume_checkpoint() -> Path:
    return Path(RESUME_CHECKPOINT)


def build_datasets(config: Dict[str, Any]):
    from raevdm.data import RLEDClipDataset

    train_cfg = dict(config["data"])
    train_cfg["split"] = "train"
    train_set = RLEDClipDataset(**train_cfg)

    val_cfg = dict(config["data"])
    val_cfg["split"] = "val"
    val_cfg["random_crop"] = False
    val_cfg["random_flip"] = False
    try:
        val_set = RLEDClipDataset(**val_cfg)
    except Exception as exc:
        print(f"[warn] could not build val split ({exc}); using all split center-crop data for validation.")
        val_cfg["split"] = "all"
        val_set = RLEDClipDataset(**val_cfg)
    return train_set, val_set


def make_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr: float, base_lr: float):
    import torch

    min_ratio = min(max(float(min_lr) / max(float(base_lr), 1e-12), 0.0), 1.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-6)
        denom = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def load_warm_start_weights(model, checkpoint_path: Path, device) -> Dict[str, Any]:
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Residual warm-start checkpoint not found. Set "
            f"RAE_RESUME_CHECKPOINT to a valid core checkpoint: {checkpoint_path}"
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_key = "model_ema" if LOAD_EMA_WEIGHTS and "model_ema" in ckpt else "model"
    checkpoint_state = ckpt[state_key]
    model_state = model.state_dict()
    loaded = 0
    expanded = 0
    skipped = []
    patched_state = {}
    for key, target in model_state.items():
        source = checkpoint_state.get(key)
        if source is None:
            patched_state[key] = target
            skipped.append((key, "missing"))
            continue
        if tuple(source.shape) == tuple(target.shape):
            patched_state[key] = source
            loaded += 1
            continue
        if key == "denoiser.input.weight" and source.ndim == target.ndim and source.shape[0] == target.shape[0]:
            new_weight = target.clone()
            new_weight.zero_()
            channels = min(source.shape[1], target.shape[1])
            new_weight[:, :channels].copy_(source[:, :channels])
            patched_state[key] = new_weight
            expanded += 1
            continue
        patched_state[key] = target
        skipped.append((key, f"shape {tuple(source.shape)} -> {tuple(target.shape)}"))
    model.load_state_dict(patched_state, strict=True)
    model.to(device)
    print(
        f"Loaded warm-start weights from {checkpoint_path} ({state_key}), "
        f"epoch={ckpt.get('epoch')}, step={ckpt.get('step')}, "
        f"loaded={loaded}, expanded={expanded}, skipped={len(skipped)}"
    )
    if skipped:
        preview = ", ".join([f"{name}:{reason}" for name, reason in skipped[:8]])
        print(f"Warm-start skipped preview: {preview}")
    return ckpt


def update_ema(ema_model, model, decay: float) -> None:
    import torch

    with torch.no_grad():
        model_state = unwrap_model(model).state_dict()
        ema_state = ema_model.state_dict()
        for key, value in ema_state.items():
            source = model_state[key].detach()
            if torch.is_floating_point(value):
                value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                value.copy_(source)


def validate(model, loader, device, max_batches: int) -> Dict[str, float]:
    from raevdm.utils import to_device

    was_training = model.training
    model.eval()
    sums = {
        "model_middle_psnr": 0.0,
        "model_middle_mae": 0.0,
        "linear_middle_psnr": 0.0,
        "linear_middle_mae": 0.0,
        "gain_psnr": 0.0,
        "gain_mae": 0.0,
    }
    count = 0
    with __import__("torch").no_grad():
        for i, batch in enumerate(loader):
            if i >= int(max_batches):
                break
            batch = to_device(batch, device)
            out = residual_forward(model, batch)
            pred = bcthw_to_btchw(out["pred"])
            linear = bcthw_to_btchw(out["linear"])
            gt = batch["frames"].float()
            key_mask = batch["keyframe_mask"].float()

            model_metrics = middle_metrics_btchw(pred, gt, key_mask)
            linear_metrics = middle_metrics_btchw(linear, gt, key_mask)
            sums["model_middle_psnr"] += model_metrics["middle_psnr"]
            sums["model_middle_mae"] += model_metrics["middle_mae"]
            sums["linear_middle_psnr"] += linear_metrics["middle_psnr"]
            sums["linear_middle_mae"] += linear_metrics["middle_mae"]
            sums["gain_psnr"] += model_metrics["middle_psnr"] - linear_metrics["middle_psnr"]
            sums["gain_mae"] += linear_metrics["middle_mae"] - model_metrics["middle_mae"]
            count += 1
    if was_training:
        model.train()
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    metrics["middle_psnr_gain_vs_linear"] = metrics.pop("gain_psnr")
    metrics["middle_mae_gain_vs_linear"] = metrics.pop("gain_mae")
    return metrics


def save_payload(path: Path, model, optimizer, epoch: int, step: int, config: Dict[str, Any], extra: Dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "config": config,
    }
    payload.update(extra)
    torch.save(payload, path)


def worker(local_rank: int, world_size: int, config: Dict[str, Any]) -> None:
    add_src_to_path()
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from tqdm import tqdm

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = local_rank
    is_main = rank == 0
    set_seed(int(config.get("seed", 3407)) + rank)

    run_dir = Path(config["output_dir"]) / config["run_name"]
    resume_checkpoint = resolve_resume_checkpoint()
    if is_main:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        save_config(config, run_dir / "config_residual.yaml")
        print(f"Run dir: {run_dir}")
        print(f"World size: {world_size}")
        print(f"Data root: {config['data']['root']}")
        print(f"Residual warm-start checkpoint: {resume_checkpoint}")
        print(f"Residual correction config: {config['residual_correction']}")
    safe_barrier(distributed, local_rank)

    train_set, val_set = build_datasets(config)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else None
    train_loader_kwargs = {
        "batch_size": int(config["train"]["batch_size"]),
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": int(config["train"]["num_workers"]),
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    if int(config["train"]["num_workers"]) > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = int(config["train"].get("prefetch_factor", 4))
    train_loader = DataLoader(train_set, **train_loader_kwargs)
    val_loader = DataLoader(
        val_set,
        batch_size=int(VAL_BATCH_SIZE),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config)
    load_warm_start_weights(model, resume_checkpoint, device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    ema_decay = float(config["train"].get("ema_decay", 0.0))
    ema_model = None
    if ema_decay > 0:
        ema_model = copy.deepcopy(unwrap_model(model)).to(device)
        ema_model.eval()
        set_requires_grad(ema_model, False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", WEIGHT_DECAY)),
    )
    total_steps = max(len(train_loader), 1) * int(config["train"]["epochs"])
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(config["train"].get("warmup_steps", 0)),
        min_lr=float(config["train"].get("min_lr", 0.0)),
        base_lr=float(config["train"]["lr"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["mixed_precision"]) and device.type == "cuda")

    if is_main:
        print(f"Train clips: {len(train_set)}")
        print(f"Val clips: {len(val_set)}")
        print(f"Parameters: {count_parameters(unwrap_model(model)) / 1e6:.2f} M")
        print(f"Batch size per GPU: {config['train']['batch_size']}")
        print(f"Global batch size: {int(config['train']['batch_size']) * world_size}")
        print(f"Loss weights: {LOSS_WEIGHTS}")

    global_step = 0
    best_middle_psnr = -1e9
    no_improve = 0

    if is_main:
        val_model = ema_model if ema_model is not None else unwrap_model(model)
        initial_metrics = validate(val_model, val_loader, device, VAL_MAX_BATCHES)
        print(f"initial val={initial_metrics}")
        extra = {
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "finetune_metrics": initial_metrics,
            "residual_correction": dict(config["residual_correction"]),
        }
        if ema_model is not None:
            extra["model_ema"] = ema_model.state_dict()
        save_payload(run_dir / "checkpoints" / "initial.pt", model, optimizer, -1, global_step, config, extra)
        save_payload(run_dir / "checkpoints" / "best_middle.pt", model, optimizer, -1, global_step, config, extra)
        best_middle_psnr = float(initial_metrics.get("model_middle_psnr", -1e9))
        print(f"initial best middle PSNR: {best_middle_psnr:.4f}")
    safe_barrier(distributed, local_rank)

    for epoch in range(int(config["train"]["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        start = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"residual epoch {epoch:03d}", dynamic_ncols=True, disable=not is_main)
        for _, batch in enumerate(pbar):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=bool(config["train"]["mixed_precision"]) and device.type == "cuda"):
                outputs = residual_forward(model, batch)
                total, losses = compute_residual_losses(batch, outputs)

            scaler.scale(total).backward()
            if float(config["train"].get("grad_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            if new_scale >= old_scale:
                scheduler.step()
            if ema_model is not None:
                update_ema(ema_model, model, ema_decay)

            global_step += 1
            epoch_loss += float(total.detach().cpu())
            epoch_count += 1
            if is_main and (global_step % int(config["train"].get("log_every", 50)) == 0):
                loss_text = " ".join(f"{k}:{float(v.detach().cpu()):.4f}" for k, v in losses.items())
                pbar.set_postfix_str(loss_text)

        safe_barrier(distributed, local_rank)

        stop_now = False
        if is_main:
            val_model = ema_model if ema_model is not None else unwrap_model(model)
            val_metrics = {}
            if (epoch + 1) % int(VAL_EVERY) == 0:
                val_metrics = validate(val_model, val_loader, device, VAL_MAX_BATCHES)
            avg_loss = epoch_loss / max(epoch_count, 1)
            elapsed = time.perf_counter() - start
            print(f"epoch {epoch:03d} loss={avg_loss:.5f} elapsed={elapsed/60:.1f}min val={val_metrics}")

            extra = {
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "finetune_metrics": val_metrics,
                "residual_correction": dict(config["residual_correction"]),
            }
            if ema_model is not None:
                extra["model_ema"] = ema_model.state_dict()
            save_payload(run_dir / "checkpoints" / "latest.pt", model, optimizer, epoch, global_step, config, extra)
            if (epoch + 1) % int(config["train"].get("save_every", 1)) == 0:
                save_payload(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, global_step, config, extra)

            score = float(val_metrics.get("model_middle_psnr", -1e9))
            if score > best_middle_psnr + 1e-4:
                best_middle_psnr = score
                no_improve = 0
                save_payload(run_dir / "checkpoints" / "best_middle.pt", model, optimizer, epoch, global_step, config, extra)
                print(f"new best middle PSNR: {best_middle_psnr:.4f}")
            else:
                no_improve += 1
                print(f"no improvement for {no_improve}/{EARLY_STOP_PATIENCE} epochs")
                if no_improve >= int(EARLY_STOP_PATIENCE):
                    stop_now = True
                    print("early stopping: validation did not improve.")

        if distributed:
            flag = __import__("torch").tensor([1 if stop_now else 0], device=device, dtype=__import__("torch").int32)
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

    import torch
    import torch.multiprocessing as mp

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested = [x.strip() for x in GPU_IDS.split(",") if x.strip()]
    world_size = len(requested) if USE_DDP and cuda_count >= len(requested) and len(requested) > 1 else 1
    if world_size > 1:
        mp.spawn(worker, nprocs=world_size, args=(world_size, config), join=True)
    else:
        worker(0, 1, config)


if __name__ == "__main__":
    main()
