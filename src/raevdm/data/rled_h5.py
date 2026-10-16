from __future__ import annotations

import bisect
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .event_voxel import build_event_representation


@dataclass
class SequenceMeta:
    path: Path
    image_keys: List[str]
    timestamps: np.ndarray
    event_indices: np.ndarray
    height: int
    width: int
    num_events: int


def _read_attr_or_dataset(node: h5py.Dataset | h5py.Group, name: str, default: Any = None) -> Any:
    if name in node.attrs:
        return node.attrs[name]
    if isinstance(node, h5py.Group) and name in node:
        return node[name][()]
    return default


def _image_keys(images_group: h5py.Group) -> List[str]:
    return sorted(list(images_group.keys()))


def read_image_node(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        image = node[()]
    else:
        candidates = [k for k in node.keys() if isinstance(node[k], h5py.Dataset)]
        if not candidates:
            raise ValueError("Image group does not contain an image dataset.")
        image = node[sorted(candidates)[0]][()]
    if image.ndim == 3:
        image = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(image.dtype)
    return image


def inspect_sequence(path: str | Path) -> SequenceMeta:
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "events" not in f or "images" not in f:
            raise ValueError(f"{path} is not an RLED-style H5 file.")
        keys = _image_keys(f["images"])
        if not keys:
            raise ValueError(f"{path} has no images.")

        timestamps = []
        event_indices = []
        for key in keys:
            node = f["images"][key]
            timestamps.append(float(_read_attr_or_dataset(node, "timestamp", np.nan)))
            event_indices.append(int(_read_attr_or_dataset(node, "event_idx", -1)))

        first = f["images"][keys[0]]
        size = _read_attr_or_dataset(first, "size", None)
        if size is not None:
            height, width = int(size[0]), int(size[1])
        else:
            img = read_image_node(first)
            height, width = int(img.shape[0]), int(img.shape[1])

        num_events = len(f["events/ts"] if "ts" in f["events"] else f["events/t"])
    return SequenceMeta(
        path=path,
        image_keys=keys,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        event_indices=np.asarray(event_indices, dtype=np.int64),
        height=height,
        width=width,
        num_events=num_events,
    )


def discover_h5_files(root: str | Path, pattern: str = "**/*.h5", max_files: Optional[int] = None) -> List[Path]:
    root = Path(root)
    files = sorted(root.glob(pattern))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No H5 files found under {root} with pattern {pattern}.")
    return files


class RLEDClipDataset(Dataset):
    """RLED H5 dataset that returns sparse-keyframe event-conditioned clips."""

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        file_pattern: str = "**/*.h5",
        max_files: Optional[int] = None,
        clip_len: int = 5,
        clip_stride: int = 1,
        frame_step: int = 1,
        event_bins: int = 3,
        crop_size: Optional[Sequence[int]] = (192, 192),
        random_crop: bool = True,
        random_flip: bool = True,
        keyframe_policy: str = "endpoints",
        keyframe_stride: int = 4,
        contrast_threshold: float | Sequence[float] = 0.2,
        val_every: int = 10,
        event_cache_size: int = 16,
    ) -> None:
        self.root = Path(root)
        self.clip_len = int(clip_len)
        self.clip_stride = int(max(clip_stride, 1))
        self.frame_step = int(max(frame_step, 1))
        self.event_bins = int(max(event_bins, 1))
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.random_crop = bool(random_crop)
        self.random_flip = bool(random_flip)
        self.keyframe_policy = keyframe_policy
        self.keyframe_stride = int(max(keyframe_stride, 1))
        threshold = np.asarray(contrast_threshold, dtype=np.float32)
        if threshold.shape not in {(), (2,)} or not np.isfinite(threshold).all() or (threshold <= 0).any():
            raise ValueError("contrast_threshold must be positive: scalar or [C+,C-].")
        self.contrast_threshold = float(threshold) if threshold.ndim == 0 else tuple(map(float, threshold))
        self.event_cache_size = int(max(event_cache_size, 0))
        self._h5_cache: Dict[str, h5py.File] = {}
        self._event_cache: OrderedDict[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

        files = discover_h5_files(self.root, file_pattern, max_files)
        if split in {"train", "val"}:
            selected = []
            for i, path in enumerate(files):
                is_val = (i % int(max(val_every, 1))) == 0
                if (split == "val" and is_val) or (split == "train" and not is_val):
                    selected.append(path)
            files = selected
        self.metas = [inspect_sequence(p) for p in files]
        self.samples: List[Tuple[int, int]] = []
        for seq_idx, meta in enumerate(self.metas):
            max_start = len(meta.image_keys) - 1 - (self.clip_len - 1) * self.frame_step
            for start in range(0, max_start + 1, self.clip_stride):
                self.samples.append((seq_idx, start))
        if not self.samples:
            raise RuntimeError("No valid clips were constructed. Check clip_len and RLED files.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5_cache"] = {}
        state["_event_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_h5_cache", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def __getitem__(self, index: int) -> Dict[str, Any]:
        seq_idx, start = self.samples[index]
        meta = self.metas[seq_idx]
        frame_indices = [start + i * self.frame_step for i in range(self.clip_len)]

        crop = self._sample_crop(meta.height, meta.width)
        flip = self.random_flip and random.random() < 0.5

        f = self._get_h5(meta.path)
        frames = [self._read_frame(f, meta.image_keys[i], crop, flip) for i in frame_indices]
        events = [self._read_interval_events(f, meta, frame_indices[i], frame_indices[i + 1], crop, flip) for i in range(self.clip_len - 1)]

        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        keyframes, keyframe_mask = self._make_sparse_keyframes(frames_np)

        cond_list, raw_list, count_list, rel_list, obs_list = [], [], [], [], []
        delta_list, precision_list, interval_obs_list = [], [], []
        for ev in events:
            cond_list.append(ev["condition"])
            raw_list.append(ev["raw"])
            count_list.append(ev["counts"])
            rel_list.append(ev["reliability"])
            obs_list.append(ev["observability"])
            delta_list.append(ev["event_delta"])
            precision_list.append(ev["interval_precision"])
            interval_obs_list.append(ev["interval_observability"])

        event_condition = np.concatenate(cond_list, axis=1)
        event_raw = np.concatenate(raw_list, axis=1)
        event_counts = np.concatenate(count_list, axis=1)
        event_prior_reliability = np.concatenate(rel_list, axis=1)
        event_prior_observability = np.concatenate(obs_list, axis=1)

        sample = {
            "frames": torch.from_numpy(frames_np[:, None]),
            "keyframes": torch.from_numpy(keyframes[:, None]),
            "keyframe_mask": torch.from_numpy(keyframe_mask[:, None]),
            "event_condition": torch.from_numpy(event_condition),
            "event_raw": torch.from_numpy(event_raw),
            "event_counts": torch.from_numpy(event_counts),
            "event_prior_reliability": torch.from_numpy(event_prior_reliability),
            "event_prior_observability": torch.from_numpy(event_prior_observability),
            "event_delta": torch.from_numpy(np.stack(delta_list, axis=0)[:, None]),
            "event_interval_precision": torch.from_numpy(np.stack(precision_list, axis=0)[:, None]),
            "event_interval_observability": torch.from_numpy(np.stack(interval_obs_list, axis=0)[:, None]),
            "timestamps": torch.from_numpy(meta.timestamps[frame_indices].astype(np.float64)),
            "path": str(meta.path),
            "start_index": int(start),
            "event_bins": int(self.event_bins),
            "frame_step": int(self.frame_step),
            "contrast_threshold": torch.tensor(self.contrast_threshold, dtype=torch.float32),
        }
        return sample

    def _get_h5(self, path: Path) -> h5py.File:
        key = str(path)
        handle = self._h5_cache.get(key)
        if handle is None:
            handle = h5py.File(path, "r")
            self._h5_cache[key] = handle
        return handle

    def _sample_crop(self, height: int, width: int) -> Tuple[int, int, int, int]:
        if self.crop_size is None:
            return 0, 0, height, width
        crop_h = min(int(self.crop_size[0]), height)
        crop_w = min(int(self.crop_size[1]), width)
        if self.random_crop:
            top = random.randint(0, max(height - crop_h, 0))
            left = random.randint(0, max(width - crop_w, 0))
        else:
            top = max((height - crop_h) // 2, 0)
            left = max((width - crop_w) // 2, 0)
        return top, left, crop_h, crop_w

    def _read_frame(self, f: h5py.File, key: str, crop: Tuple[int, int, int, int], flip: bool) -> np.ndarray:
        image = read_image_node(f["images"][key])
        top, left, crop_h, crop_w = crop
        image = image[top : top + crop_h, left : left + crop_w]
        if flip:
            image = np.ascontiguousarray(image[:, ::-1])
        return image

    def _event_range(self, f: h5py.File, meta: SequenceMeta, idx0: int, idx1: int) -> Tuple[int, int, float, float]:
        event_indices = meta.event_indices
        if event_indices[idx0] >= 0 and event_indices[idx1] >= 0:
            start = int(event_indices[idx0])
            end = int(event_indices[idx1])
        else:
            ts_dset = f["events/ts"] if "ts" in f["events"] else f["events/t"]
            start = bisect.bisect_left(ts_dset, meta.timestamps[idx0])
            end = bisect.bisect_left(ts_dset, meta.timestamps[idx1])
        start = max(0, min(start, meta.num_events))
        end = max(start, min(end, meta.num_events))
        return start, end, float(meta.timestamps[idx0]), float(meta.timestamps[idx1])

    def _read_interval_events(
        self,
        f: h5py.File,
        meta: SequenceMeta,
        idx0: int,
        idx1: int,
        crop: Tuple[int, int, int, int],
        flip: bool,
    ) -> Dict[str, np.ndarray]:
        start, end, t0, t1 = self._event_range(f, meta, idx0, idx1)
        xs, ys, ts, ps = self._get_interval_arrays(f, meta.path, start, end)

        top, left, crop_h, crop_w = crop
        valid = (xs >= left) & (xs < left + crop_w) & (ys >= top) & (ys < top + crop_h)
        xs = xs[valid].astype(np.int64) - left
        ys = ys[valid].astype(np.int64) - top
        ts = ts[valid]
        ps = ps[valid]
        if flip:
            xs = crop_w - 1 - xs

        rep = build_event_representation(
            xs=xs,
            ys=ys,
            ts=ts,
            ps=ps,
            height=crop_h,
            width=crop_w,
            bins=self.event_bins,
            t0=t0,
            t1=t1,
            contrast_threshold=self.contrast_threshold,
        )
        return {
            "condition": rep.condition,
            "raw": rep.raw,
            "reliability": rep.reliability,
            "observability": rep.observability,
            "event_delta": rep.event_delta,
            "interval_precision": rep.interval_precision,
            "interval_observability": rep.interval_observability,
            "counts": rep.counts,
        }

    def _get_interval_arrays(
        self,
        f: h5py.File,
        path: Path,
        start: int,
        end: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_key = (str(path), int(start), int(end))
        cached = self._event_cache.get(cache_key)
        if cached is not None:
            self._event_cache.move_to_end(cache_key)
            return cached

        xs_key = "xs" if "xs" in f["events"] else "x"
        ys_key = "ys" if "ys" in f["events"] else "y"
        ts_key = "ts" if "ts" in f["events"] else "t"
        ps_key = "ps" if "ps" in f["events"] else "p"

        xs = f[f"events/{xs_key}"][start:end]
        ys = f[f"events/{ys_key}"][start:end]
        ts = f[f"events/{ts_key}"][start:end]
        ps = f[f"events/{ps_key}"][start:end]

        arrays = (xs, ys, ts, ps)
        if self.event_cache_size > 0:
            self._event_cache[cache_key] = arrays
            while len(self._event_cache) > self.event_cache_size:
                self._event_cache.popitem(last=False)
        return arrays

    def _make_sparse_keyframes(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.zeros_like(frames, dtype=np.float32)
        if self.keyframe_policy == "endpoints":
            keep = [0, len(frames) - 1]
        elif self.keyframe_policy == "stride":
            keep = list(range(0, len(frames), self.keyframe_stride))
            if len(frames) - 1 not in keep:
                keep.append(len(frames) - 1)
        elif self.keyframe_policy == "all":
            keep = list(range(len(frames)))
        else:
            raise ValueError(f"Unknown keyframe_policy: {self.keyframe_policy}")
        mask[keep] = 1.0
        keyframes = frames * mask
        return keyframes.astype(np.float32), mask.astype(np.float32)
