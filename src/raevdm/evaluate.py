from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import RLEDClipDataset
from .metrics import interpolation_metrics
from .runtime import load_runtime, predict, predict_tiled
from .utils import to_device


def make_dataset(config):
    data = dict(config["data"])
    data.update(split="all", random_crop=False, random_flip=False, crop_size=None, keyframe_policy="endpoints")
    return RLEDClipDataset(**data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sequence-balanced evaluation; unknown-frame fidelity metrics.")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="Official test split root, or native-resolution HQF H5 root.")
    parser.add_argument("--raft-weights", default=os.environ.get("RAE_RAFT_WEIGHTS"))
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--lpips-weights", help="Optional full LPIPS-Alex state_dict including its feature backbone.")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, raft, transforms = load_runtime(args.config, args.checkpoint, device,
                                                  args.raft_weights, not args.no_ema)
    config["data"].update(root=args.data_root, frame_step=args.frame_step, max_files=None)
    loader = DataLoader(make_dataset(config), batch_size=1, num_workers=int(config.get("eval", {}).get("num_workers", 0)))
    perceptual = None
    if args.lpips_weights:
        import lpips
        perceptual = lpips.LPIPS(net="alex", pretrained=False, pnet_rand=True).to(device).eval()
        perceptual.load_state_dict(torch.load(args.lpips_weights, map_location="cpu", weights_only=True), strict=True)
    sequences, clip_results = {}, []
    with torch.inference_mode():
        for index, batch in enumerate(tqdm(loader, desc="evaluation")):
            if args.max_clips is not None and index >= args.max_clips:
                break
            if args.tiled:
                prediction = predict_tiled(model, batch, config, device, raft, transforms, args.tile_size, args.overlap)
            else:
                prediction = predict(model, to_device(batch, device), config, raft, transforms).cpu()
            metrics = interpolation_metrics(prediction, batch["frames"])
            if perceptual is not None:
                p = prediction[:, 1:-1].flatten(0, 1).repeat(1, 3, 1, 1).to(device) * 2 - 1
                g = batch["frames"][:, 1:-1].flatten(0, 1).repeat(1, 3, 1, 1).to(device) * 2 - 1
                metrics["lpips"] = float(perceptual(p, g).mean())
            sequence = str(Path(batch["path"][0]).relative_to(Path(args.data_root)))
            sequences.setdefault(sequence, []).append(metrics)
            clip_results.append(dict(sequence=sequence, start_index=int(batch["start_index"][0]), **metrics))
    if not sequences:
        raise RuntimeError("No test clips were evaluated.")
    means = {name: {metric: sum(row[metric] for row in rows) / len(rows) for metric in rows[0]}
             for name, rows in sequences.items()}
    overall = {key: sum(row[key] for row in means.values()) / len(means) for key in next(iter(means.values()))}
    report = dict(protocol=dict(frame_step=args.frame_step, keyframes="two endpoints",
                                mode="deterministic", tiled=args.tiled,
                                tile_size=args.tile_size if args.tiled else None,
                                overlap=args.overlap if args.tiled else None,
                                averaging="within each sequence, then across sequences",
                                lpips_computed=perceptual is not None),
                  overall=overall, sequences=means, clips=clip_results)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
