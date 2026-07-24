from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Folder containing RLED H5 files.")
    parser.add_argument("--pattern", default="**/*.h5")
    args = parser.parse_args()

    files = sorted(Path(args.root).glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No H5 files found under {args.root}.")
    path = files[0]
    print(f"file: {path}")
    with h5py.File(path, "r") as f:
        print("root:", list(f.keys()))
        print("events:", list(f["events"].keys()))
        keys = sorted(list(f["images"].keys()))
        print("num_images:", len(keys))
        print("first_images:", keys[:3])
        for key in keys[:3]:
            node = f["images"][key]
            print(key, "shape=", getattr(node, "shape", None), "attrs=", dict(node.attrs))
        if "event_indices" in f:
            print("event_indices:", f["event_indices"].shape)


if __name__ == "__main__":
    main()
