"""Exercise the full architecture without an optimizer update or dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from raevdm.flow_warp import synthesize_occlusion_aware_warp_video
from raevdm.flow_warp import estimate_bidirectional_flow, load_frozen_raft_small
from raevdm.losses import compute_losses
from raevdm.models import build_model
from raevdm.stages import activate_phase, configure_stage, optimizer_groups, phase_at_epoch, phase_weights, stage_forward
from raevdm.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--output", default="outputs/implementation_check.json")
    parser.add_argument("--raft-weights", help="Optional local RAFT-Small weights; use --size 128 or larger.")
    args = parser.parse_args()
    torch.manual_seed(3407)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    config = load_config(ROOT / "configs/paper.yaml")
    model = build_model(configure_stage(config, "refinement")).to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model))
    frames = torch.rand(1, 7, 1, args.size, args.size, device=device) * 0.7 + 0.15
    mask = torch.zeros_like(frames)
    mask[:, (0, -1)] = 1
    counts = torch.randint(0, 4, (1, 2, 30, args.size, args.size), device=device).float()
    batch = dict(frames=frames, keyframes=frames * mask, keyframe_mask=mask,
                 event_counts=counts, event_raw=counts / 3, event_bins=5, contrast_threshold=0.2)
    zero = torch.zeros(1, 2, args.size, args.size, device=device)
    geometry = "Analytic zero-flow fixture; not a real-RAFT accuracy benchmark."
    flow01 = flow10 = zero
    if args.raft_weights:
        raft, transforms = load_frozen_raft_small(device, args.raft_weights)
        assert all(not p.requires_grad for p in raft.parameters())
        flow01, flow10 = estimate_bidirectional_flow(raft, transforms, frames[:, 0], frames[:, -1])
        geometry = "Frozen pretrained RAFT-Small on synthetic anchors; not a dataset accuracy benchmark."
    flow = synthesize_occlusion_aware_warp_video(frames[:, 0], frames[:, -1], flow01, flow10, 7)
    checks = []
    for stage, epoch in [("pretrain", 0), ("residual", 0), ("flow", 0), ("flow", 3),
                         ("refinement", 0), ("refinement", 3), ("refinement", 12)]:
        stage_config = configure_stage(config, stage)
        phase = phase_at_epoch(stage_config, epoch)
        activate_phase(optimizer, phase)
        weights = phase_weights(stage_config, phase)
        model.zero_grad(set_to_none=True)
        outputs = stage_forward(model, batch, stage_config, weights, flow, phase)
        loss, _ = compute_losses(batch, outputs, weights, 5, theory_cfg=config["theory"])
        if not torch.isfinite(loss):
            raise AssertionError("Non-finite forward loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError("Missing or non-finite gradients")
        checks.append(dict(stage=stage, phase=phase["name"], finite_forward_and_backward=True,
                           gradient_tensors=len(gradients)))
    model.eval()
    calls = []
    hook = model.denoiser.register_forward_hook(lambda *_: calls.append(1))
    observed = {k: v for k, v in batch.items() if k != "frames"}
    prediction = model.reconstruct(observed, flow, config["residual"])
    hook.remove()
    assert len(calls) == 1
    torch.testing.assert_close(prediction[:, (0, -1)], frames[:, (0, -1)], rtol=0, atol=0)
    result = dict(device=str(device), parameters=sum(p.numel() for p in model.parameters()),
                  shape=list(prediction.shape), phases=checks, deterministic_predictor_calls=len(calls),
                  anchors_exact=True, optimizer_updates=0, trained_weights_written=False,
                  geometry=geometry)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
