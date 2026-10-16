import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from raevdm.data import RLEDClipDataset
from raevdm.event_statistics import event_confidence_at_events
from raevdm.metrics import temporal_error
from raevdm.models import build_model
from raevdm.runtime import load_runtime


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example_h5(tmp_path):
    path = tmp_path / "sequence.h5"
    times = np.arange(19, dtype=np.float64) / 30
    events = np.linspace(0.0, times[-1], 900, endpoint=False)
    with h5py.File(path, "w") as handle:
        handle["events/xs"] = np.arange(900) % 20
        handle["events/ys"] = np.arange(900) % 16
        handle["events/ts"] = events
        handle["events/ps"] = np.arange(900) % 2
        for index, time in enumerate(times):
            image = np.full((16, 20), 70 + index * 3, np.uint8)
            node = handle.create_dataset(f"images/image{index:09d}", data=image)
            node.attrs["timestamp"] = time
            node.attrs["event_idx"] = np.searchsorted(events, time)
    return path


def test_dataset_stride_and_polarity_threshold_collation(example_h5):
    dataset = RLEDClipDataset(example_h5.parent, split="all", clip_len=7, frame_step=3,
                              event_bins=5, crop_size=None, random_flip=False,
                              contrast_threshold=[0.3, 0.1])
    sample = dataset[0]
    assert sample["frames"].shape == (7, 1, 16, 20)
    assert sample["event_counts"].shape == (2, 30, 16, 20)
    np.testing.assert_allclose(sample["timestamps"], np.arange(0, 19, 3) / 30)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    assert batch["contrast_threshold"].shape == (1, 2)
    assert sample["keyframe_mask"][1:-1].sum() == 0


def test_event_score_uses_same_soft_temporal_assignment():
    maps = torch.tensor([0.2, 0.8, 0.4, 0.6, 0.1, 0.9]).reshape(2, 3, 1, 1)
    scores = event_confidence_at_events(maps, [0, 0], [0, 0], [0.25, 0.75], [1, -1], [0, 1], 3)
    torch.testing.assert_close(scores, torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError, match="half-open"):
        event_confidence_at_events(maps, [0], [0], [1.0], [1], [0, 1], 3)


def test_temporal_error_is_not_temporal_smoothing():
    gt = torch.tensor([0.1, 0.5, 0.9]).reshape(1, 3, 1, 1, 1)
    assert temporal_error(gt, gt) == 0
    assert temporal_error(torch.ones_like(gt) * 0.5, gt).item() == pytest.approx(0.4)


def test_cli_inference_and_sequence_evaluation(example_h5, tmp_path, tiny_config):
    checkpoint = tmp_path / "untrained_test_fixture.pt"
    tiny_config["implementation_version"] = "tcsvt-20261016"
    torch.save(dict(model=build_model(tiny_config).state_dict(), config=tiny_config), checkpoint)
    out = tmp_path / "predictions"
    infer = [sys.executable, "infer.py", "--config", "configs/train_core.yaml", "--checkpoint", str(checkpoint),
             "--h5", str(example_h5), "--output-dir", str(out), "--frame-step", "3"]
    subprocess.run(infer, cwd=ROOT, check=True, capture_output=True, text=True, timeout=90)
    assert len(list((out / "sample_0000_pred").glob("*.png"))) == 7
    report = tmp_path / "evaluation.json"
    evaluate = [sys.executable, "evaluate.py", "--checkpoint", str(checkpoint), "--data-root", str(example_h5.parent),
                "--frame-step", "3", "--max-clips", "1", "--output", str(report)]
    subprocess.run(evaluate, cwd=ROOT, check=True, capture_output=True, text=True, timeout=90)
    result = json.loads(report.read_text())
    assert result["protocol"]["mode"] == "deterministic"
    assert len(result["sequences"]) == 1
    assert set(result["overall"]) == {"psnr", "ssim", "mae", "temporal_error"}


def test_legacy_checkpoint_uses_its_own_architecture_defaults(tmp_path, tiny_config):
    tiny_config["model"].pop("reliability_cue_channels")
    model = build_model(tiny_config)
    path = tmp_path / "legacy_test_fixture.pt"
    torch.save(dict(model=model.state_dict(), config={"model": tiny_config["model"],
                                                    "diffusion": tiny_config["diffusion"]}), path)
    with pytest.warns(UserWarning, match="Legacy checkpoint"):
        restored, _, _, _ = load_runtime(ROOT / "configs/inference.yaml", path, torch.device("cpu"))
    assert restored.event_encoder.reliability.net[0].in_channels == 2


def test_dataset_inspection_entrypoints(example_h5):
    commands = [[sys.executable, "tools/check_event_representation.py", "--h5", str(example_h5)],
                [sys.executable, "tools/inspect_rled_h5.py", "--root", str(example_h5.parent)]]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=45)
