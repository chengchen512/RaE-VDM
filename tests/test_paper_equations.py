import numpy as np
import pytest
import torch

from raevdm.data.event_voxel import build_event_representation, events_to_neg_pos_voxels, reliability_prior_from_counts
from raevdm.event_statistics import count_priors, learned_event_statistics, signed_event_delta
from raevdm.losses import compute_losses, log_intensity_delta


def test_temporal_voxels_half_open_and_mass_conservation():
    pos, neg = events_to_neg_pos_voxels(np.array([0, 0, 0, 0]), np.zeros(4),
                                       np.array([-0.1, 0.25, 0.75, 1.0]),
                                       np.array([1, 1, -1, 1]), 1, 1, 3, 0.0, 1.0)
    np.testing.assert_allclose(pos[:, 0, 0], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(neg[:, 0, 0], [0.0, 0.5, 0.5])
    assert pos.sum() + neg.sum() == 2


def test_count_priors_match_independent_numpy_implementation():
    counts = torch.randint(0, 15, (2, 2, 6, 5, 7)).float()
    actual = count_priors(counts, 3)
    for sample in range(2):
        for interval in range(2):
            pos, neg = counts[sample, :, interval * 3:(interval + 1) * 3].numpy()
            rp, rn, quiet = reliability_prior_from_counts(pos, neg)
            np.testing.assert_allclose(actual["count_support"][sample, :, interval * 3:(interval + 1) * 3],
                                       np.stack([rp, rn]), atol=3e-7)
            np.testing.assert_allclose(actual["count_quiet"][sample, 0, interval], quiet.mean(0), atol=3e-7)


def test_empty_stream_has_zero_support_and_unit_silence():
    prior = count_priors(torch.zeros(1, 2, 6, 5, 7), 3)
    assert torch.count_nonzero(prior["count_support"]) == 0
    assert torch.all(prior["count_silence"] == 1)
    assert torch.count_nonzero(prior["count_motion"]) == 0


def test_raw_count_delta_support_and_active_bin_average():
    counts = torch.zeros(1, 2, 3, 1, 1)
    counts[0, 0, :, 0, 0] = torch.tensor([1., 3., 0.])
    rel = torch.full_like(counts, 0.9)
    rel[0, 0, :, 0, 0] = torch.tensor([0.2, 0.8, 0.9])
    out = learned_event_statistics(counts, rel, torch.ones(1, 1, 3, 1, 1), 3, 0.2)
    assert out["learned_mass"].item() == pytest.approx(2.6)
    assert out["learned_support"].item() == pytest.approx(2.6 / 3.6)
    assert out["learned_delta"].item() == pytest.approx(0.52)
    assert out["supported_confidence"].item() == pytest.approx(0.5, abs=1e-6)
    # Count-domain physical targets are not clipped to [-2,2].
    assert signed_event_delta(counts * 100, rel, 3).item() > 2


def test_polarity_specific_thresholds():
    counts = torch.ones(2, 2, 3, 1, 1)
    threshold = torch.tensor([[0.3, 0.1], [0.4, 0.2]])
    out = signed_event_delta(counts, counts, 3, threshold)
    torch.testing.assert_close(out.flatten(), torch.tensor([0.6, 0.6]))


def test_detached_weights_and_confidence_targets(batch):
    pred = batch["frames"].permute(0, 2, 1, 3, 4).clone().requires_grad_()
    rel = torch.full_like(batch["event_counts"], 0.5, requires_grad=True)
    obs = torch.full((1, 1, 6, 12, 16), 0.7, requires_grad=True)
    out = dict(pred=pred, pred_raw=pred, event=dict(reliability=rel, observability=obs))
    _, losses = compute_losses(batch, out, {"event_physics": 1.0}, 3)
    grads = torch.autograd.grad(losses["quiet"], [pred, rel, obs], allow_unused=True, retain_graph=True)
    assert grads[0] is not None and grads[1] is None and grads[2] is None
    grads = torch.autograd.grad(losses["event_physics"], [rel, obs], allow_unused=True, retain_graph=True)
    assert grads[0].abs().sum() > 0 and grads[1] is None
    grads = torch.autograd.grad(losses["reliability_calibration"], [pred, rel], allow_unused=True)
    assert grads[0] is None and grads[1].abs().sum() > 0


def test_confidence_loss_ignores_unsupported_intervals(batch):
    batch["event_counts"] = torch.zeros_like(batch["event_counts"])
    pred = batch["frames"].permute(0, 2, 1, 3, 4)
    rel = torch.ones_like(batch["event_counts"], requires_grad=True)
    obs = rel[:, :1]
    _, losses = compute_losses(batch, dict(pred=pred, event=dict(reliability=rel, observability=obs)),
                               {"reliability_supervision": 1.0}, 3)
    assert losses["reliability_calibration"].item() == 0
    assert losses["reliability_supervision"].item() == pytest.approx(0.01)


def test_log_operator_adds_epsilon_in_dark_pixels():
    x = torch.tensor([0.0, 0.1]).reshape(1, 1, 2, 1, 1)
    assert log_intensity_delta(x).item() == pytest.approx(np.log(0.1001) - np.log(0.0001))
