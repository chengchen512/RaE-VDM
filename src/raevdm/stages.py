"""Four-stage optimization contract in Sec. IV-A."""
from __future__ import annotations

import copy

import torch

from .flow_warp import estimate_bidirectional_flow, synthesize_occlusion_aware_warp_video


STAGES = ("pretrain", "residual", "flow", "refinement")
CHECKPOINT_NAMES = {
    "pretrain": "best_core.pt", "residual": "best_middle.pt",
    "flow": "best_flow_fusion.pt", "refinement": "FINAL_BEST_INFERENCE.pt",
}


def configure_stage(config: dict, stage: str) -> dict:
    if stage not in STAGES:
        raise ValueError(f"Unknown training stage: {stage}")
    config = copy.deepcopy(config)
    config["train"]["stage"] = stage
    config["model"]["use_flow_fusion"] = stage in {"flow", "refinement"}
    config["model"]["use_occlusion_refinement"] = stage == "refinement"
    config["model"]["event_gate_source"] = "count"
    mode = {"pretrain": "diffusion", "residual": "residual_correction",
            "flow": "flow_fusion", "refinement": "occlusion_refinement"}[stage]
    config["finetune"] = {"mode": mode}
    return config


def phase_at_epoch(config: dict, epoch: int) -> dict:
    phases = config["training_stages"][config["train"]["stage"]]
    offset = 0
    for phase in phases:
        if offset <= epoch < offset + int(phase["epochs"]):
            return dict(phase)
        offset += int(phase["epochs"])
    raise ValueError(f"Epoch {epoch} lies outside the configured stage.")


def phase_weights(config: dict, phase: dict) -> dict:
    if not phase.get("reconstruction_losses", True):
        return {"diffusion": float(phase.get("diffusion", 0.0))}
    weights = dict(config["loss"])
    weights["diffusion"] = float(phase.get("diffusion", 0.0))
    weights["reliability_supervision"] = float(phase.get("confidence", 0.0))
    weights["anchor"] = float(phase.get("anchor", 0.0))
    for key, value in config.get("loss_overrides", {}).items():
        weights[key] = float(value)
    if phase.get("backbone_lr", 0.0) == 0:
        weights["diffusion"] = 0.0
        weights["reliability_supervision"] = 0.0
    return weights


def optimizer_groups(model) -> list[dict]:
    groups = [{"name": "backbone", "params": list(model.event_encoder.parameters())
               + list(model.denoiser.parameters()), "lr": 0.0}]
    for name, module in (("flow", model.flow_fusion), ("refinement", model.occlusion_refinement)):
        if module is not None:
            groups.append({"name": name, "params": list(module.parameters()), "lr": 0.0})
    return groups


def activate_phase(optimizer, phase: dict) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(phase.get(group["name"] + "_lr", 0.0))
        for parameter in group["params"]:
            parameter.requires_grad_(group["lr"] > 0)


@torch.no_grad()
def geometry_inputs(batch: dict, raft, transforms, updates: int = 12) -> dict:
    anchors = batch["keyframes"].float()
    flow01, flow10 = estimate_bidirectional_flow(raft, transforms, anchors[:, 0], anchors[:, -1], updates)
    result = synthesize_occlusion_aware_warp_video(
        anchors[:, 0], anchors[:, -1], flow01, flow10, anchors.shape[1],
        timestamps=batch.get("timestamps"),
    )
    return {**result, "flow01": flow01, "flow10": flow10}


def stage_forward(model, batch: dict, config: dict, weights: dict,
                  flow_inputs: dict | None = None, phase: dict | None = None):
    stage = config["train"]["stage"]
    if stage == "pretrain":
        return model(batch, mode=config["train"].get("pretraining_objective", "diffusion"),
                     residual_cfg=config.get("residual", {}))
    mode = {"residual": "supervised_residual", "flow": "flow_fusion",
            "refinement": "occlusion_refinement"}[stage]
    return model(batch, mode=mode, flow_inputs=flow_inputs,
                 residual_cfg=config.get("residual", {}),
                 base_raft_peak=float(config.get("flow_fusion", {}).get("base_raft_peak", 0.55)),
                 detach_base=bool(phase is not None and phase.get("backbone_lr", 0.0) == 0),
                 include_diffusion=float(weights.get("diffusion", 0.0)) > 0)


def load_stage_weights(model, checkpoint: dict, allow_new_heads: bool = False) -> str:
    key = "model_ema" if "model_ema" in checkpoint else "model"
    state = checkpoint[key]
    current = model.state_dict()
    unexpected = set(state) - set(current)
    missing = set(current) - set(state)
    mismatched = [name for name in set(state) & set(current) if state[name].shape != current[name].shape]
    allowed_missing = all(name.startswith(("flow_fusion.", "occlusion_refinement.")) for name in missing)
    if unexpected or mismatched or (missing and not (allow_new_heads and allowed_missing)):
        raise ValueError(f"Checkpoint architecture mismatch: missing={sorted(missing)}, "
                         f"unexpected={sorted(unexpected)}, shape_mismatch={mismatched}. "
                         "Use a matching architecture/config; no tensors were silently discarded.")
    model.load_state_dict(state, strict=not allow_new_heads)
    return key
