from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch


@pytest.fixture(autouse=True)
def reproducible_cpu_threads():
    torch.manual_seed(17)
    torch.set_num_threads(2)


@pytest.fixture
def batch():
    frames = torch.rand(1, 3, 1, 12, 16) * 0.6 + 0.2
    mask = torch.zeros_like(frames)
    mask[:, (0, -1)] = 1
    counts = torch.randint(0, 4, (1, 2, 6, 12, 16)).float()
    return dict(frames=frames, keyframes=frames * mask, keyframe_mask=mask,
                event_counts=counts, event_raw=counts / 3,
                event_bins=torch.tensor([3]), timestamps=torch.tensor([[0., 0.3, 1.]], dtype=torch.float64),
                contrast_threshold=torch.tensor([0.2]))


@pytest.fixture
def tiny_config():
    from raevdm.utils import load_config
    config = load_config(Path(__file__).resolve().parents[1] / "configs/paper.yaml")
    config["model"].update(event_feature_channels=4, base_channels=4, channel_mult=[1, 2],
                           time_embed_dim=16, flow_fusion_hidden_channels=4,
                           occlusion_refinement_hidden_channels=4)
    config["data"].update(event_bins=3, clip_len=3)
    config["diffusion"]["timesteps"] = 10
    return config
