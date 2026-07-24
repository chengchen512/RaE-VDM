from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import RLEDClipDataset
from .models import build_model
from .utils import load_config, to_device


def save_clip(video: torch.Tensor, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = (video.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    gif_frames = []
    for i, frame in enumerate(frames):
        img = frame[0] if frame.ndim == 3 else frame
        imageio.imwrite(out_dir / f"{prefix}_{i:03d}.png", img)
        gif_frames.append(img)
    imageio.mimsave(out_dir / f"{prefix}.gif", gif_frames, duration=0.08)


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
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
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
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(config).to(device)
    state_key = "model" if args.no_ema or "model_ema" not in ckpt else "model_ema"
    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()
    print(f"loaded checkpoint weights: {state_key}")

    sampling_steps = int(args.sampling_steps or config["diffusion"].get("sampling_steps", 50))
    out_dir = Path(args.out_dir or Path(config["output_dir"]) / config["run_name"] / "samples")
    for i, batch in enumerate(tqdm(loader, desc="infer", dynamic_ncols=True)):
        if i >= args.num_samples:
            break
        batch = to_device(batch, device)
        pred = model.sample(batch, sampling_steps=sampling_steps)
        gt = batch["frames"]
        save_clip(pred[0], out_dir, f"sample_{i:03d}_pred")
        save_clip(gt[0], out_dir, f"sample_{i:03d}_gt")
    print(f"samples saved to: {out_dir}")


if __name__ == "__main__":
    main()
