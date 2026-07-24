from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np

event_voxel_path = Path(__file__).resolve().parents[1] / "src" / "raevdm" / "data" / "event_voxel.py"
spec = importlib.util.spec_from_file_location("event_voxel", event_voxel_path)
event_voxel = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(event_voxel)
build_event_representation = event_voxel.build_event_representation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--frame0", type=int, default=0)
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=64)
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as f:
        keys = sorted(f["images"].keys())
        k0, k1 = args.frame0, args.frame0 + 1
        img0 = f["images"][keys[k0]]
        img1 = f["images"][keys[k1]]
        h, w = map(int, img0.attrs["size"][:2])
        crop_h = crop_w = min(args.crop_size, h, w)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

        start = int(img0.attrs["event_idx"])
        end = int(img1.attrs["event_idx"])
        xs = f["events/xs"][start:end]
        ys = f["events/ys"][start:end]
        ts = f["events/ts"][start:end]
        ps = f["events/ps"][start:end]

        valid = (xs >= left) & (xs < left + crop_w) & (ys >= top) & (ys < top + crop_h)
        xs = xs[valid] - left
        ys = ys[valid] - top
        ts = ts[valid]
        ps = ps[valid]

        rep = build_event_representation(
            xs,
            ys,
            ts,
            ps,
            crop_h,
            crop_w,
            args.bins,
            float(img0.attrs["timestamp"]),
            float(img1.attrs["timestamp"]),
        )

    print("condition:", rep.condition.shape, rep.condition.dtype, float(rep.condition.min()), float(rep.condition.max()))
    print("raw:", rep.raw.shape, "reliability:", rep.reliability.shape, "observability:", rep.observability.shape)
    print("event_delta:", rep.event_delta.shape, float(rep.event_delta.min()), float(rep.event_delta.max()))
    print("interval_precision:", float(rep.interval_precision.mean()))
    print("interval_observability:", float(rep.interval_observability.mean()))


if __name__ == "__main__":
    main()
