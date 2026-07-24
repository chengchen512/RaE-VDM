from __future__ import annotations

import argparse
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import RLEDClipDataset
from .metrics import compute_basic_metrics
from .models import build_model
from .utils import load_config, to_device


def make_dataset(config: Dict[str, Any]) -> RLEDClipDataset:
    data_cfg = dict(config["data"])
    data_cfg["split"] = "val" if data_cfg.get("split", "all") == "train" else data_cfg.get("split", "all")
    data_cfg["random_crop"] = False
    data_cfg["random_flip"] = False
    return RLEDClipDataset(**data_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA weights.")
    args = parser.parse_args()

    config = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = ckpt.get("config")
    if isinstance(checkpoint_config, dict):
        if "model" in checkpoint_config:
            config["model"] = checkpoint_config["model"]
        if "diffusion" in checkpoint_config:
            config["diffusion"] = checkpoint_config["diffusion"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = make_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=int(config["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["eval"]["num_workers"]),
    )

    model = build_model(config).to(device)
    state_key = "model" if args.no_ema or "model_ema" not in ckpt else "model_ema"
    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()
    print(f"loaded checkpoint weights: {state_key}")

    sampling_steps = int(args.sampling_steps or config["diffusion"].get("sampling_steps", 50))
    sums = {"psnr": 0.0, "mae": 0.0}
    count = 0
    max_batches = config["eval"].get("max_batches")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="eval", dynamic_ncols=True)):
            if max_batches is not None and i >= int(max_batches):
                break
            batch = to_device(batch, device)
            pred = model.sample(batch, sampling_steps=sampling_steps)
            metrics = compute_basic_metrics(pred, batch["frames"].float())
            for key in sums:
                sums[key] += metrics[key]
            count += 1
    for key, value in sums.items():
        print(f"{key}: {value / max(count, 1):.4f}")


if __name__ == "__main__":
    main()
