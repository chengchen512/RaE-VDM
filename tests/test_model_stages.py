import copy

import pytest
import torch

from raevdm.flow_warp import synthesize_occlusion_aware_warp_video
from raevdm.losses import compute_losses
from raevdm.models import build_model
from raevdm.models.raevdm import linear_keyframe_interpolation
from raevdm.runtime import observation_batch, predict, predict_tiled
from raevdm.stages import (activate_phase, configure_stage, load_stage_weights,
                           optimizer_groups, phase_at_epoch, phase_weights, stage_forward)


def zero_flow(batch):
    image0, image1 = batch["keyframes"][:, 0], batch["keyframes"][:, -1]
    flow = image0.new_zeros(image0.shape[0], 2, *image0.shape[-2:])
    return synthesize_occlusion_aware_warp_video(image0, image1, flow, flow, image0.shape[0] * 0 + 3,
                                                timestamps=batch["timestamps"])


def test_nonuniform_target_timestamps(batch):
    base = linear_keyframe_interpolation(batch["keyframes"], batch["keyframe_mask"], batch["timestamps"])
    torch.testing.assert_close(base[:, 1], 0.7 * batch["frames"][:, 0] + 0.3 * batch["frames"][:, -1])
    warp = zero_flow(batch)
    torch.testing.assert_close(warp["time_map"][:, :, 1], torch.full((1, 1, 12, 16), 0.3))


def test_gate_depends_only_on_counts_and_has_floor(batch, tiny_config):
    model = build_model(tiny_config)
    a = model.event_gate(batch, 3)
    altered = {**batch, "event_raw": torch.rand_like(batch["event_raw"]),
               "event_interval_observability": torch.rand(1, 2, 1, 12, 16)}
    b = model.event_gate(altered, 3, event_out={"reliability": torch.zeros_like(batch["event_raw"])})
    torch.testing.assert_close(a, b)
    empty = {**batch, "event_counts": torch.zeros_like(batch["event_counts"])}
    gate = model.event_gate(empty, 3)
    assert torch.all(gate[:, :, 1] == 0.08)
    assert torch.count_nonzero(gate[:, :, (0, -1)]) == 0


def test_deterministic_prediction_single_call_without_gt(batch, tiny_config):
    model = build_model(tiny_config).eval()
    calls = []
    hook = model.denoiser.register_forward_hook(lambda *_: calls.append(1))
    first = predict(model, batch, tiny_config)
    assert len(calls) == 1
    second = predict(model, {key: value for key, value in batch.items() if key != "frames"}, tiny_config)
    hook.remove()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first[:, (0, -1)], batch["frames"][:, (0, -1)], rtol=0, atol=0)
    assert first.min() >= 0 and first.max() <= 1


@pytest.mark.parametrize("stage,epoch", [("pretrain", 0), ("residual", 0), ("flow", 0),
                                         ("flow", 3), ("refinement", 0), ("refinement", 3), ("refinement", 12)])
def test_each_stage_phase_has_finite_forward_and_backward(batch, tiny_config, stage, epoch):
    config = configure_stage(tiny_config, stage)
    model = build_model(config)
    optimizer = torch.optim.AdamW(optimizer_groups(model))
    phase = phase_at_epoch(config, epoch)
    activate_phase(optimizer, phase)
    weights = phase_weights(config, phase)
    outputs = stage_forward(model, batch, config, weights, zero_flow(batch), phase)
    total, losses = compute_losses(batch, outputs, weights, 3, theory_cfg=config["theory"])
    assert torch.isfinite(total)
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    if phase.get("backbone_lr", 0) == 0:
        assert all(p.grad is None for p in model.denoiser.parameters())


def test_preprojection_anchor_loss_reaches_refinement_head(batch, tiny_config):
    model = build_model(configure_stage(tiny_config, "refinement"))
    with torch.no_grad():
        model.occlusion_refinement.out.bias[4] = 0.3
    out = model(batch, mode="occlusion_refinement", flow_inputs=zero_flow(batch))
    _, losses = compute_losses(batch, out, {"anchor": 0.1}, 3)
    assert losses["anchor"] > 0
    losses["anchor"].backward()
    assert model.occlusion_refinement.out.bias.grad[4].abs() > 0
    torch.testing.assert_close(out["pred"][:, :, (0, -1)],
                               batch["frames"].permute(0, 2, 1, 3, 4)[:, :, (0, -1)], rtol=0, atol=0)


def test_stage_schedule_and_loss_coefficients(tiny_config):
    assert [sum(p["epochs"] for p in tiny_config["training_stages"][s])
            for s in ("pretrain", "residual", "flow", "refinement")] == [120, 20, 15, 15]
    config = configure_stage(tiny_config, "refinement")
    head = phase_weights(config, phase_at_epoch(config, 0))
    final = phase_weights(config, phase_at_epoch(config, 12))
    assert head["diffusion"] == head["reliability_supervision"] == 0
    assert final["diffusion"] == 0.05 and final["reliability_supervision"] == 1
    assert final["anchor"] == 0.1 and final["temporal"] == 0.005


def test_checkpoint_warm_start_only_allows_new_geometry_heads(tiny_config):
    source = build_model(tiny_config)
    target = build_model(configure_stage(tiny_config, "refinement"))
    load_stage_weights(target, {"model": source.state_dict()}, allow_new_heads=True)
    bad = dict(source.state_dict())
    bad.pop("denoiser.input.weight")
    with pytest.raises(ValueError, match="architecture mismatch"):
        load_stage_weights(target, {"model": bad}, allow_new_heads=True)


def test_tiled_small_native_frame_no_resize_or_missing_pixels(batch, tiny_config):
    model = build_model(tiny_config).eval()
    direct = predict(model, batch, tiny_config)
    tiled = predict_tiled(model, batch, tiny_config, torch.device("cpu"), tile_size=32, overlap=8)
    torch.testing.assert_close(direct, tiled)
    overlapping = predict_tiled(model, batch, tiny_config, torch.device("cpu"), tile_size=8, overlap=4)
    assert overlapping.shape == direct.shape and torch.isfinite(overlapping).all()
    torch.testing.assert_close(overlapping[:, (0, -1)], batch["frames"][:, (0, -1)], rtol=0, atol=0)
