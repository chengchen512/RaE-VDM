from __future__ import annotations

import copy
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


# =================================
# Flow-fusion fine-tune setup
# =================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "configs" / "stage_base.yaml"
DATA_ROOT = os.environ.get("RAE_DATA_ROOT", str(PROJECT_DIR / "data" / "RLED"))

RESUME_CHECKPOINT = Path(os.environ.get("RAE_RESUME_CHECKPOINT", PROJECT_DIR / "best_middle.pt"))
RAFT_WEIGHTS_PATH = Path(
    os.environ.get("RAE_RAFT_WEIGHTS", PROJECT_DIR / "raft_small_C_T_V2-01064c6d.pth")
)
REQUIRE_LOCAL_RAFT_WEIGHTS = True

GPU_IDS = os.environ.get("RAE_GPU_IDS", "0,1")
USE_DDP = True
MASTER_PORT = "29651"

RUN_NAME_PREFIX = os.environ.get("RAE_RUN_PREFIX", "raevdm_rled_flow_fusion_final_from_residual_best")
OUTPUT_DIR = Path(os.environ.get("RAE_OUTPUT_DIR", PROJECT_DIR / "outputs"))

# Stage 1 learns only the zero-initialized fusion behavior. Stage 2 jointly
# fine-tunes the existing residual model with a very small learning rate.
EPOCHS = 15
HEAD_ONLY_EPOCHS = 3
EARLY_STOP_PATIENCE = 6
FUSION_LR = 1e-4
BASE_LR = 2e-6
MIN_LR_RATIO = 0.10
WARMUP_STEPS = 500
WEIGHT_DECAY = 1e-6
EMA_DECAY = 0.9995

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
    "recon": 6.0,
    "gradient": 0.25,
    "temporal": 0.20,
    "best_branch_guard": 4.0,
    "residual_guard": 2.0,
    "oracle_alpha": 0.20,
    "refinement_l1": 0.10,
    "alpha_tv": 0.05,
}
CHARBONNIER_EPS = 1e-3
BRANCH_GUARD_MARGIN = 0.002
ORACLE_ALPHA_TEMPERATURE = 0.03

VAL_EVERY = 1
VAL_BATCH_SIZE = 2
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
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad)


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


def temporal_delta_loss(pred, gt, mask, eps: float):
    pred_delta = pred[:, :, 1:] - pred[:, :, :-1]
    gt_delta = gt[:, :, 1:] - gt[:, :, :-1]
    delta_mask = mask[:, :, 1:] * mask[:, :, :-1]
    return masked_mean(charbonnier_tensor(pred_delta - gt_delta, eps), delta_mask)


def alpha_tv_loss(alpha, mask, eps: float):
    dx = alpha[..., :, 1:] - alpha[..., :, :-1]
    dx_mask = mask[..., :, 1:] * mask[..., :, :-1]
    dy = alpha[..., 1:, :] - alpha[..., :-1, :]
    dy_mask = mask[..., 1:, :] * mask[..., :-1, :]
    dt = alpha[:, :, 1:] - alpha[:, :, :-1]
    dt_mask = mask[:, :, 1:] * mask[:, :, :-1]
    return (
        masked_mean(charbonnier_tensor(dx, eps), dx_mask)
        + masked_mean(charbonnier_tensor(dy, eps), dy_mask)
        + 0.5 * masked_mean(charbonnier_tensor(dt, eps), dt_mask)
    ) / 2.5


def compute_fusion_losses(outputs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    pred = outputs["pred"]
    gt = outputs["gt"]
    residual = outputs["residual_pred"]
    warped = outputs["warped_video"]
    alpha = outputs["alpha"]
    refinement = outputs["refinement"]
    mask = (1.0 - outputs["keyframe_mask"]).clamp(0.0, 1.0)

    recon = masked_mean(charbonnier_tensor(pred - gt, CHARBONNIER_EPS), mask)
    gradient = spatial_gradient_loss(pred, gt, mask, CHARBONNIER_EPS)
    temporal = temporal_delta_loss(pred, gt, mask, CHARBONNIER_EPS)

    pred_err = (pred - gt).abs()
    residual_err = (residual - gt).abs().detach()
    warped_err = (warped - gt).abs().detach()
    best_branch_err = torch.minimum(residual_err, warped_err)
    best_branch_guard = masked_mean(
        charbonnier_tensor(F.relu(pred_err - best_branch_err - BRANCH_GUARD_MARGIN), CHARBONNIER_EPS),
        mask,
    )
    residual_guard = masked_mean(
        charbonnier_tensor(F.relu(pred_err - residual_err - BRANCH_GUARD_MARGIN), CHARBONNIER_EPS),
        mask,
    )

    oracle_alpha_target = torch.sigmoid(
        (residual_err - warped_err) / float(ORACLE_ALPHA_TEMPERATURE)
    ).detach()
    alpha_fp32 = alpha.float().clamp(1e-4, 1.0 - 1e-4)
    target_fp32 = oracle_alpha_target.float()
    oracle_alpha = -target_fp32 * torch.log(alpha_fp32) - (1.0 - target_fp32) * torch.log1p(-alpha_fp32)
    oracle_alpha = masked_mean(oracle_alpha, mask)

    refinement_l1 = masked_mean(charbonnier_tensor(refinement, CHARBONNIER_EPS), mask)
    alpha_tv = alpha_tv_loss(alpha, mask, CHARBONNIER_EPS)

    losses = {
        "recon": recon,
        "gradient": gradient,
        "temporal": temporal,
        "best_branch_guard": best_branch_guard,
        "residual_guard": residual_guard,
        "oracle_alpha": oracle_alpha,
        "refinement_l1": refinement_l1,
        "alpha_tv": alpha_tv,
    }
    total = pred.new_tensor(0.0)
    for name, value in losses.items():
        total = total + float(LOSS_WEIGHTS.get(name, 0.0)) * value
    losses["total"] = total
    return total, losses


def middle_metrics(pred_btchw, gt_btchw, key_mask_btchw) -> Dict[str, float]:
    import torch

    mask = (1.0 - key_mask_btchw.float()).clamp(0.0, 1.0)
    err = (pred_btchw.float() - gt_btchw.float()) * mask
    denom = mask.sum().clamp_min(1.0)
    mae = err.abs().sum() / denom
    mse = (err * err).sum() / denom
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return {
        "middle_mae": float(mae.detach().cpu()),
        "middle_psnr": float(psnr.detach().cpu()),
    }


def resolve_resume_checkpoint() -> Path:
    return Path(RESUME_CHECKPOINT)


def build_config() -> Dict[str, Any]:
    import torch

    from raevdm.utils import load_config

    config = load_config(CONFIG_PATH)
    source = torch.load(resolve_resume_checkpoint(), map_location="cpu", weights_only=False)
    source_config = source.get("config", {})
    for section in ("model", "diffusion", "theory", "residual"):
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
            "flow_fusion_hidden_channels": 32,
            "flow_fusion_refinement_scale": 0.05,
            "flow_fusion_alpha_delta_scale": 2.0,
        }
    )
    config["model"] = model_cfg

    train = dict(config["train"])
    train.update(
        {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "epochs": EPOCHS,
            "fusion_lr": FUSION_LR,
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
        "mode": "flow_fusion",
        "resume_checkpoint": str(resolve_resume_checkpoint()),
        "head_only_epochs": HEAD_ONLY_EPOCHS,
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
    }
    return config


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
        return float(MIN_LR_RATIO) + (1.0 - float(MIN_LR_RATIO)) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def load_residual_warm_start(model, checkpoint_path: Path, device) -> Dict[str, Any]:
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Residual best checkpoint not found. Put it at:\n"
            f"  {RESUME_CHECKPOINT}"
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_mode = ckpt.get("config", {}).get("finetune", {}).get("mode")
    if source_mode != "residual_correction":
        raise RuntimeError(
            "The warm-start checkpoint is not the residual best model. "
            f"Expected finetune.mode='residual_correction', got {source_mode!r}."
        )
    state_key = "model_ema" if "model_ema" in ckpt else "model"
    source_state = ckpt[state_key]
    target_state = model.state_dict()
    loaded = 0
    expected_new = []
    unexpected = []
    patched = {}
    for key, target in target_state.items():
        source = source_state.get(key)
        if source is not None and tuple(source.shape) == tuple(target.shape):
            patched[key] = source
            loaded += 1
        elif key.startswith("flow_fusion."):
            patched[key] = target
            expected_new.append(key)
        else:
            patched[key] = target
            unexpected.append(key)
    if unexpected:
        raise RuntimeError(f"Unexpected warm-start mismatches: {unexpected[:12]}")
    model.load_state_dict(patched, strict=True)
    if float(model.flow_fusion.out.weight.detach().abs().sum()) != 0.0:
        raise RuntimeError("Flow-fusion output layer is not zero initialized.")
    model.to(device)
    print(
        f"Loaded residual warm-start from {checkpoint_path} ({state_key}), "
        f"epoch={ckpt.get('epoch')}, step={ckpt.get('step')}, "
        f"loaded={loaded}, new_flow_fusion_tensors={len(expected_new)}"
    )
    return ckpt


def build_flow_inputs(raft, transforms, batch: Dict[str, Any], flow_updates: int) -> Dict[str, Any]:
    from raevdm.flow_warp import estimate_bidirectional_flow, synthesize_bidirectional_warp_video

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
    warped, confidence, time_map = synthesize_bidirectional_warp_video(
        image0,
        image1,
        flow01,
        flow10,
        frames_t=batch["frames"].shape[1],
    )
    return {
        "warped_video": warped.detach(),
        "warp_confidence": confidence.detach(),
        "time_map": time_map.detach(),
        "flow01": flow01.detach(),
        "flow10": flow10.detach(),
    }


def forward_fusion_model(
    model,
    batch: Dict[str, Any],
    flow_inputs: Dict[str, Any],
    detach_base: bool,
) -> Dict[str, Any]:
    return model(
        batch,
        mode="flow_fusion",
        flow_inputs=flow_inputs,
        residual_cfg=RESIDUAL_CFG,
        base_raft_peak=BASE_RAFT_PEAK,
        detach_base=detach_base,
    )


def update_ema(ema_model, model, decay: float) -> None:
    import torch

    with torch.no_grad():
        source_state = unwrap_model(model).state_dict()
        ema_state = ema_model.state_dict()
        for key, value in ema_state.items():
            source = source_state[key].detach()
            if torch.is_floating_point(value):
                value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                value.copy_(source)


def validate(model, raft, transforms, loader, device, max_batches: int) -> Dict[str, float]:
    from raevdm.utils import to_device

    was_training = model.training
    model.eval()
    sums = {
        "fusion_middle_psnr": 0.0,
        "fusion_middle_mae": 0.0,
        "residual_middle_psnr": 0.0,
        "residual_middle_mae": 0.0,
        "warp_middle_psnr": 0.0,
        "warp_middle_mae": 0.0,
        "linear_middle_psnr": 0.0,
        "linear_middle_mae": 0.0,
    }
    count = 0
    with __import__("torch").no_grad():
        for index, batch in enumerate(loader):
            if index >= int(max_batches):
                break
            batch = to_device(batch, device)
            flow_inputs = build_flow_inputs(raft, transforms, batch, FLOW_UPDATES_VAL)
            outputs = forward_fusion_model(model, batch, flow_inputs, detach_base=False)
            pred = bcthw_to_btchw(outputs["pred"])
            residual = bcthw_to_btchw(outputs["residual_pred"])
            warped = bcthw_to_btchw(outputs["warped_video"])
            linear = bcthw_to_btchw(outputs["linear"])
            gt = batch["frames"].float()
            key_mask = batch["keyframe_mask"].float()

            for prefix, video in (
                ("fusion", pred),
                ("residual", residual),
                ("warp", warped),
                ("linear", linear),
            ):
                metrics = middle_metrics(video, gt, key_mask)
                sums[f"{prefix}_middle_psnr"] += metrics["middle_psnr"]
                sums[f"{prefix}_middle_mae"] += metrics["middle_mae"]
            count += 1

    if was_training:
        model.train()
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    metrics["psnr_gain_vs_residual"] = metrics["fusion_middle_psnr"] - metrics["residual_middle_psnr"]
    metrics["psnr_gain_vs_linear"] = metrics["fusion_middle_psnr"] - metrics["linear_middle_psnr"]
    metrics["mae_gain_vs_residual"] = metrics["residual_middle_mae"] - metrics["fusion_middle_mae"]
    return metrics


def save_payload(path: Path, model, optimizer, epoch: int, step: int, config: Dict[str, Any], extra: Dict[str, Any]) -> None:
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


def preflight(config: Dict[str, Any]) -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if not resolve_resume_checkpoint().exists():
        raise FileNotFoundError(f"Residual best checkpoint not found: {resolve_resume_checkpoint()}")
    if REQUIRE_LOCAL_RAFT_WEIGHTS and not RAFT_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            "Bundled RAFT weight not found. Upload this file beside the script:\n"
            f"  {RAFT_WEIGHTS_PATH}"
        )
    if not Path(DATA_ROOT).exists():
        raise FileNotFoundError(f"Training data root not found: {DATA_ROOT}")
    try:
        from torchvision.models.optical_flow import raft_small  # noqa: F401
    except Exception as exc:
        raise RuntimeError("torchvision RAFT is unavailable in RaE_env.") from exc
    print("Preflight passed.")
    print(f"Residual checkpoint: {resolve_resume_checkpoint()}")
    print(f"RAFT weights: {RAFT_WEIGHTS_PATH}")
    print(f"Run name: {config['run_name']}")


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = local_rank == 0
    set_seed(int(config.get("seed", 3407)) + local_rank)
    run_dir = Path(config["output_dir"]) / config["run_name"]
    if is_main:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        save_config(config, run_dir / "config_flow_fusion.yaml")
        print(f"Run dir: {run_dir}")
        print(f"World size: {world_size}")
        print(f"Data root: {DATA_ROOT}")
    safe_barrier(distributed, local_rank)

    train_set, val_set = build_datasets(config)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=True) if distributed else None
    train_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    if NUM_WORKERS > 0:
        train_kwargs["persistent_workers"] = True
        train_kwargs["prefetch_factor"] = PREFETCH_FACTOR
    train_loader = DataLoader(train_set, **train_kwargs)
    val_loader = DataLoader(
        val_set,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config)
    load_residual_warm_start(model, resolve_resume_checkpoint(), device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    raft, raft_transforms = load_frozen_raft_small(device, RAFT_WEIGHTS_PATH)

    fusion_parameters = list(unwrap_model(model).flow_fusion.parameters())
    fusion_ids = {id(parameter) for parameter in fusion_parameters}
    base_parameters = [parameter for parameter in model.parameters() if id(parameter) not in fusion_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": fusion_parameters, "lr": FUSION_LR, "name": "fusion"},
            {"params": base_parameters, "lr": BASE_LR, "name": "base"},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = max(len(train_loader), 1) * EPOCHS
    scheduler = make_scheduler(optimizer, total_steps, WARMUP_STEPS)
    scaler = torch.amp.GradScaler("cuda", enabled=MIXED_PRECISION and device.type == "cuda")

    ema_model = copy.deepcopy(unwrap_model(model)).to(device).eval()
    set_requires_grad(ema_model, False)

    if is_main:
        print(f"Train clips: {len(train_set)}")
        print(f"Val clips: {len(val_set)}")
        print(f"Total parameters: {count_parameters(unwrap_model(model)) / 1e6:.2f} M")
        print(f"Fusion parameters: {sum(p.numel() for p in fusion_parameters) / 1e6:.3f} M")
        print(f"Global batch size: {BATCH_SIZE * world_size}")
        print(f"Head-only epochs: {HEAD_ONLY_EPOCHS}")
        print(f"Loss weights: {LOSS_WEIGHTS}")

    global_step = 0
    best_psnr = -1e9
    no_improve = 0

    if is_main:
        initial_metrics = validate(ema_model, raft, raft_transforms, val_loader, device, VAL_MAX_BATCHES)
        print(f"initial val={initial_metrics}")
        extra = {
            "model_ema": ema_model.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "finetune_metrics": initial_metrics,
            "flow_fusion": dict(config["flow_fusion"]),
        }
        save_payload(run_dir / "checkpoints" / "initial.pt", model, optimizer, -1, 0, config, extra)
        save_payload(run_dir / "checkpoints" / "best_middle.pt", model, optimizer, -1, 0, config, extra)
        save_payload(run_dir / "checkpoints" / "best_flow_fusion.pt", model, optimizer, -1, 0, config, extra)
        best_psnr = float(initial_metrics["fusion_middle_psnr"])
        print(f"initial best fusion middle PSNR: {best_psnr:.4f}")
    safe_barrier(distributed, local_rank)

    for epoch in range(EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        detach_base = epoch < HEAD_ONLY_EPOCHS
        epoch_loss = 0.0
        epoch_count = 0
        start = time.perf_counter()
        stage = "head" if detach_base else "joint"
        pbar = tqdm(train_loader, desc=f"flow fusion {stage} epoch {epoch:03d}", dynamic_ncols=True, disable=not is_main)

        for batch in pbar:
            batch = to_device(batch, device)
            with torch.no_grad():
                flow_inputs = build_flow_inputs(raft, raft_transforms, batch, FLOW_UPDATES_TRAIN)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=MIXED_PRECISION and device.type == "cuda"):
                outputs = forward_fusion_model(model, batch, flow_inputs, detach_base=detach_base)
                total, losses = compute_fusion_losses(outputs)

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            update_ema(ema_model, model, EMA_DECAY)

            global_step += 1
            epoch_loss += float(total.detach().cpu())
            epoch_count += 1
            if is_main and global_step % LOG_EVERY == 0:
                text = " ".join(f"{key}:{float(value.detach().cpu()):.4f}" for key, value in losses.items())
                pbar.set_postfix_str(text)

        safe_barrier(distributed, local_rank)

        stop_now = False
        if is_main:
            metrics = validate(ema_model, raft, raft_transforms, val_loader, device, VAL_MAX_BATCHES)
            average_loss = epoch_loss / max(epoch_count, 1)
            elapsed = (time.perf_counter() - start) / 60.0
            print(f"epoch {epoch:03d} stage={stage} loss={average_loss:.5f} elapsed={elapsed:.1f}min val={metrics}")

            extra = {
                "model_ema": ema_model.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "finetune_metrics": metrics,
                "flow_fusion": dict(config["flow_fusion"]),
            }
            save_payload(run_dir / "checkpoints" / "latest.pt", model, optimizer, epoch, global_step, config, extra)
            if (epoch + 1) % SAVE_EVERY == 0:
                save_payload(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, global_step, config, extra)

            score = float(metrics["fusion_middle_psnr"])
            if score > best_psnr + 1e-4:
                best_psnr = score
                no_improve = 0
                save_payload(run_dir / "checkpoints" / "best_middle.pt", model, optimizer, epoch, global_step, config, extra)
                save_payload(run_dir / "checkpoints" / "best_flow_fusion.pt", model, optimizer, epoch, global_step, config, extra)
                print(f"new best fusion middle PSNR: {best_psnr:.4f}")
            else:
                no_improve += 1
                print(f"no improvement for {no_improve}/{EARLY_STOP_PATIENCE} epochs")
                if no_improve >= EARLY_STOP_PATIENCE:
                    stop_now = True
                    print("early stopping: flow-fusion validation stopped improving.")

        if distributed:
            flag = torch.tensor([1 if stop_now else 0], device=device, dtype=torch.int32)
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

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested = [item.strip() for item in GPU_IDS.split(",") if item.strip()]
    world_size = len(requested) if USE_DDP and len(requested) > 1 and cuda_count >= len(requested) else 1
    if world_size > 1:
        mp.spawn(worker, nprocs=world_size, args=(world_size, config), join=True)
    else:
        worker(0, 1, config)


if __name__ == "__main__":
    main()
