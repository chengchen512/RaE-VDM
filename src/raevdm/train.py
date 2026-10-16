from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .data import RLEDClipDataset
from .flow_warp import load_frozen_raft_small
from .losses import compute_losses
from .models import build_model
from .stages import (STAGES, CHECKPOINT_NAMES, activate_phase, configure_stage,
                     geometry_inputs, load_stage_weights, optimizer_groups,
                     phase_at_epoch, phase_weights, stage_forward)
from .utils import load_config, make_run_dir, save_checkpoint, set_seed, to_device


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


@torch.no_grad()
def update_ema(ema, model, decay: float):
    state = unwrap_model(model).state_dict()
    for key, value in ema.state_dict().items():
        if value.is_floating_point():
            value.mul_(decay).add_(state[key].detach(), alpha=1 - decay)
        else:
            value.copy_(state[key])


def build_dataset(config, validation=False, smoke=False):
    data = dict(config["data"])
    if validation:
        if config.get("validation_data"):
            data.update(config["validation_data"])
        else:
            data["split"] = "val"
        data.update(random_crop=False, random_flip=False)
    if smoke:
        data.update(split="all", max_files=1, crop_size=[32, 32], random_crop=False, random_flip=False)
    return RLEDClipDataset(**data)


def make_loader(dataset, config, validation=False, distributed=False, smoke=False):
    workers = 0 if smoke else int(config["train"].get("num_workers", 0))
    sampler = DistributedSampler(dataset, shuffle=True) if distributed and not validation else None
    kwargs = dict(batch_size=1 if validation or smoke else int(config["train"]["batch_size"]),
                  num_workers=workers, shuffle=not validation and sampler is None,
                  sampler=sampler, pin_memory=torch.cuda.is_available(), drop_last=False)
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=int(config["train"].get("prefetch_factor", 2)))
    return DataLoader(dataset, **kwargs), sampler


@torch.no_grad()
def validate(model, loader, config, device, raft=None, transforms=None, max_batches=20):
    model.eval()
    values = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = to_device(batch, device)
        if config["train"]["stage"] == "pretrain" and config["train"].get("pretraining_objective", "diffusion") == "diffusion":
            # Fixed RNG during validation, restored before training resumes.
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(0)
                out = model(batch, mode="diffusion")
            values.append(-float((out["noise"] - out["eps_pred"]).square().mean()))
        else:
            flow = geometry_inputs(batch, raft, transforms) if model.use_flow_fusion else None
            pred = model.reconstruct(batch, flow, config.get("residual", {}))
            mse = (pred[:, 1:-1] - batch["frames"][:, 1:-1]).square().flatten(2).mean(-1)
            values.append(float((-10 * mse.clamp_min(1e-12).log10()).mean()))
    if not values:
        raise RuntimeError("Validation contains no clips; provide validation_data explicitly.")
    return sum(values) / len(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Four-stage training from Sec. IV-A.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--resume", help="Resume this stage, including optimizer state.")
    parser.add_argument("--warm-start", help="Initialize from the previous stage; keep new heads.")
    parser.add_argument("--smoke", action="store_true", help="One tiny optimizer step for debugging only.")
    args = parser.parse_args(argv)
    if args.resume and args.warm_start:
        parser.error("Choose either --resume or --warm-start.")
    config = load_config(args.config)
    stage = args.stage or config["train"].get("stage", "pretrain")
    config = configure_stage(config, stage)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" and os.name != "nt" else "gloo")
    set_seed(int(config.get("seed", 3407)) + rank)
    if args.smoke:
        config["train"].update(max_steps_per_epoch=1, num_workers=0, mixed_precision=False)
        config["training_stages"][stage] = [{**config["training_stages"][stage][0], "epochs": 1}]
    run_dir = Path(config.get("output_dir", "outputs")) / config.get("run_name", "raevdm")
    if rank == 0:
        make_run_dir(config)
    loader, sampler = make_loader(build_dataset(config, smoke=args.smoke), config,
                                  distributed=distributed, smoke=args.smoke)
    validation = None
    if rank == 0:
        validation, _ = make_loader(build_dataset(config, validation=True, smoke=args.smoke), config,
                                    validation=True, smoke=args.smoke)
    model = build_model(config).to(device)
    checkpoint = None
    if args.resume or args.warm_start:
        checkpoint = torch.load(args.resume or args.warm_start, map_location="cpu", weights_only=True)
        if args.resume:
            if checkpoint.get("config", {}).get("train", {}).get("stage") != stage:
                raise ValueError("Use --warm-start when moving between stages.")
            model.load_state_dict(checkpoint["model"], strict=True)
        else:
            load_stage_weights(model, checkpoint, allow_new_heads=True)
    elif stage != "pretrain":
        raise ValueError("This stage needs --warm-start or --resume; do not silently train it from scratch.")
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    if args.resume and "model_ema" in checkpoint:
        ema.load_state_dict(checkpoint["model_ema"], strict=True)
    optimizer = torch.optim.AdamW(optimizer_groups(model), weight_decay=float(config["train"].get("weight_decay", 1e-6)))
    amp = bool(config["train"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, step, best = 0, 0, float("-inf")
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, step = int(checkpoint["epoch"]) + 1, int(checkpoint["step"])
        best = float(checkpoint.get("best_score", best))
    if distributed:
        # Register every parameter before changing freeze states between phases.
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    find_unused_parameters=True)
    raft = transforms = None
    if unwrap_model(model).use_flow_fusion:
        path = os.environ.get("RAE_RAFT_WEIGHTS") or config.get("flow_fusion", {}).get("raft_weights")
        if not path:
            raise ValueError("Set RAE_RAFT_WEIGHTS or flow_fusion.raft_weights to a local RAFT-Small checkpoint.")
        raft, transforms = load_frozen_raft_small(device, path)
    epochs = sum(int(p["epochs"]) for p in config["training_stages"][stage])
    previous_phase = checkpoint.get("phase") if args.resume else None
    best_path = run_dir / "checkpoints" / CHECKPOINT_NAMES[stage]
    try:
        for epoch in range(start, epochs):
            phase = phase_at_epoch(config, epoch)
            if phase["name"] != previous_phase:
                # Select within the final optimization phase of each stage.
                best = float("-inf")
                previous_phase = phase["name"]
            activate_phase(optimizer, phase)
            weights = phase_weights(config, phase)
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            for index, batch in enumerate(tqdm(loader, desc=f"{stage}:{phase['name']} {epoch}", disable=rank != 0)):
                limit = config["train"].get("max_steps_per_epoch")
                if limit is not None and index >= int(limit):
                    break
                batch = to_device(batch, device)
                flow = geometry_inputs(batch, raft, transforms,
                                       int(config.get("flow_fusion", {}).get("flow_updates_train", 12))) \
                    if raft is not None else None
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    out = stage_forward(model, batch, config, weights, flow, phase)
                    total, losses = compute_losses(batch, out, weights, int(config["data"]["event_bins"]),
                                                   float(config["loss"].get("charbonnier_eps", 1e-3)), config.get("theory"))
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Non-finite loss at {stage}, epoch {epoch}, step {step}")
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"].get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                update_ema(ema, model, float(config["train"].get("ema_decay", 0.9999)))
                step += 1
                if rank == 0 and (args.smoke or step % int(config["train"].get("log_every", 50)) == 0):
                    print({key: float(value.detach()) for key, value in losses.items()})
            if rank == 0:
                score = validate(ema, validation, config, device, raft, transforms,
                                 max_batches=1 if args.smoke else config.get("eval", {}).get("validation_batches", 20))
                improved = score > best
                best = max(best, score)
                extra = dict(model_ema=ema.state_dict(), scaler=scaler.state_dict(), phase=phase["name"], best_score=best)
                save_checkpoint(run_dir / "checkpoints/latest.pt", unwrap_model(model), optimizer, epoch, step, config, extra)
                if improved:
                    save_checkpoint(best_path, unwrap_model(model), optimizer, epoch, step, config, extra)
                print(f"Validation score: {score:.6f}; selected checkpoint: {best_path}")
            if distributed:
                dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
