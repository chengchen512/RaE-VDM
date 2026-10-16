from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import torch
from torch.utils.data import DataLoader

from .data import RLEDClipDataset
from .runtime import load_runtime, predict, predict_tiled
from .utils import to_device


def save_clip(video, out_dir, prefix):
    directory = Path(out_dir) / prefix
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(video.detach().cpu()):
        image = (frame[0].clamp(0, 1).numpy() * 255).round().astype("uint8")
        imageio.imwrite(directory / f"{index:03d}.png", image)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Single-evaluation deterministic interpolation (Sec. III-I).")
    parser.add_argument("--config", default=os.environ.get("RAE_CONFIG", "configs/inference.yaml"))
    parser.add_argument("--checkpoint", default=os.environ.get("RAE_CHECKPOINT"))
    parser.add_argument("--h5", default=os.environ.get("RAE_TEST_H5"))
    parser.add_argument("--raft-weights", default=os.environ.get("RAE_RAFT_WEIGHTS"))
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--frame-step", type=int)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args(argv)
    if not args.checkpoint:
        parser.error("Provide --checkpoint or RAE_CHECKPOINT.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, raft, transforms = load_runtime(args.config, args.checkpoint, device,
                                                  args.raft_weights, not args.no_ema)
    data = dict(config["data"])
    data.update(split="all", crop_size=None, random_crop=False, random_flip=False, keyframe_policy="endpoints")
    if args.h5:
        path = Path(args.h5)
        data.update(root=str(path.parent), file_pattern=path.name, max_files=1)
    if args.frame_step is not None:
        data["frame_step"] = args.frame_step
    loader = DataLoader(RLEDClipDataset(**data), batch_size=1, num_workers=0, shuffle=False)
    for index, batch in enumerate(loader):
        if index >= args.num_samples:
            break
        if args.tiled:
            pred = predict_tiled(model, batch, config, device, raft, transforms, args.tile_size, args.overlap)
        else:
            pred = predict(model, to_device(batch, device), config, raft, transforms).cpu()
        save_clip(pred[0], args.output_dir, f"sample_{index:04d}_pred")
    print(f"Saved deterministic predictions to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
