from __future__ import annotations

import argparse
import copy
import math
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .data import RLEDClipDataset
from .losses import compute_losses
from .models import build_model
from .utils import AverageMeter, count_parameters, load_config, make_run_dir, save_checkpoint, set_seed, to_device


def setup_distributed() -> tuple[torch.device, int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA devices.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, rank, local_rank, world_size, distributed


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def set_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    model_state = unwrap_model(model).state_dict()
    ema_state = ema_model.state_dict()
    for key, value in ema_state.items():
        source = model_state[key].detach()
        if torch.is_floating_point(value):
            value.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            value.copy_(source)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: Dict[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    name = str(train_cfg.get("scheduler", "none")).lower()
    if name in {"none", "constant", ""}:
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name}")

    epochs = int(train_cfg["epochs"])
    total_steps = int(train_cfg.get("total_steps") or max(epochs * steps_per_epoch, 1))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    base_lr = float(train_cfg["lr"])
    min_lr = float(train_cfg.get("min_lr", 0.0))
    min_ratio = min(max(min_lr / max(base_lr, 1e-12), 0.0), 1.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-6)
        denom = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_dataset(config: Dict[str, Any], smoke: bool = False) -> RLEDClipDataset:
    data_cfg = dict(config["data"])
    if smoke:
        data_cfg["max_files"] = 1
        data_cfg["crop_size"] = [64, 64]
        data_cfg["random_crop"] = False
        data_cfg["random_flip"] = False
    return RLEDClipDataset(**data_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one tiny forward/backward step.")
    args = parser.parse_args()

    device, rank, local_rank, world_size, distributed = setup_distributed()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 3407)) + rank)
    if args.smoke:
        config["train"]["epochs"] = 1
        config["train"]["max_steps_per_epoch"] = 1
        config["train"]["batch_size"] = 1
        config["train"]["mixed_precision"] = False

    if is_main_process(rank):
        run_dir = make_run_dir(config)
    else:
        run_dir = Path(config.get("output_dir", "outputs")) / config.get("run_name", "raevdm")
    if distributed:
        dist.barrier()

    dataset = build_dataset(config, smoke=args.smoke)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else None
    num_workers = int(config["train"]["num_workers"])
    loader_kwargs = {
        "batch_size": int(config["train"]["batch_size"]),
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(config["train"].get("prefetch_factor", 4))
    loader = DataLoader(dataset, **loader_kwargs)

    model = build_model(config).to(device)
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", 0))

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(config["train"].get("find_unused_parameters", False)),
        )

    ema_decay = float(config["train"].get("ema_decay", 0.0))
    ema_model = None
    if ema_decay > 0:
        ema_model = copy.deepcopy(unwrap_model(model)).to(device)
        ema_model.eval()
        set_requires_grad(ema_model, False)
        if args.resume and "ckpt" in locals() and "model_ema" in ckpt:
            ema_model.load_state_dict(ckpt["model_ema"], strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    if args.resume and "ckpt" in locals() and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    scaler = make_grad_scaler(enabled=bool(config["train"]["mixed_precision"]) and device.type == "cuda")
    if args.resume and "ckpt" in locals() and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    scheduler = build_scheduler(optimizer, config["train"], steps_per_epoch=max(len(loader), 1))
    if scheduler is not None and args.resume and "ckpt" in locals() and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    if is_main_process(rank):
        print(f"device: {device}")
        print(f"distributed: {distributed} world_size: {world_size}")
        print(f"per_gpu_batch_size: {int(config['train']['batch_size'])}")
        print(f"global_batch_size: {int(config['train']['batch_size']) * world_size}")
        print(f"clips: {len(dataset)}")
        print(f"parameters: {count_parameters(unwrap_model(model)) / 1e6:.2f} M")

    loss_weights = dict(config["loss"])
    charbonnier_eps = float(loss_weights.pop("charbonnier_eps", 1e-3))
    event_bins = int(config["data"]["event_bins"])
    max_steps = config["train"].get("max_steps_per_epoch")
    log_every = int(config["train"].get("log_every", 20))

    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        meter = AverageMeter()
        data_meter = AverageMeter()
        iter_meter = AverageMeter()
        end_time = time.perf_counter()
        pbar = tqdm(loader, desc=f"epoch {epoch:03d}", dynamic_ncols=True, disable=not is_main_process(rank))
        for step, batch in enumerate(pbar):
            data_time = time.perf_counter() - end_time
            if max_steps is not None and step >= int(max_steps):
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=bool(config["train"]["mixed_precision"]) and device.type == "cuda"):
                training_mode = str(config["train"].get("training_mode", "diffusion"))
                outputs = model(
                    batch,
                    mode=training_mode,
                    residual_cfg=dict(config.get("residual", {})),
                )
                total, losses = compute_losses(
                    batch,
                    outputs,
                    loss_weights,
                    event_bins,
                    charbonnier_eps,
                    theory_cfg=dict(config.get("theory", {})),
                )
            scaler.scale(total).backward()
            if float(config["train"].get("grad_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            if ema_model is not None:
                update_ema(ema_model, model, ema_decay)

            global_step += 1
            meter.update(float(total.detach().cpu()), n=1)
            iter_time = time.perf_counter() - end_time
            data_meter.update(data_time, n=1)
            iter_meter.update(iter_time, n=1)
            if is_main_process(rank) and (global_step % log_every == 0 or args.smoke):
                loss_text = " ".join([f"{k}:{float(v.detach().cpu()):.4f}" for k, v in losses.items()])
                pbar.set_postfix_str(f"{loss_text} data:{data_meter.avg:.2f}s iter:{iter_meter.avg:.2f}s")
            end_time = time.perf_counter()

        if is_main_process(rank):
            extra = {
                "scaler": scaler.state_dict(),
            }
            if scheduler is not None:
                extra["scheduler"] = scheduler.state_dict()
            if ema_model is not None:
                extra["model_ema"] = ema_model.state_dict()
            save_checkpoint(run_dir / "checkpoints" / "latest.pt", unwrap_model(model), optimizer, epoch, global_step, config, extra=extra)
            if (epoch + 1) % int(config["train"].get("save_every", 1)) == 0:
                save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", unwrap_model(model), optimizer, epoch, global_step, config, extra=extra)
            print(f"epoch {epoch:03d} avg_loss={meter.avg:.5f}")
        if distributed:
            dist.barrier()

    if is_main_process(rank):
        print(f"done. checkpoints: {run_dir / 'checkpoints'}")
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
