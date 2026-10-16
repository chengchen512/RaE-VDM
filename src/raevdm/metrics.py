from __future__ import annotations

import math
from typing import Dict

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred.clamp(0, max_val) - target.clamp(0, max_val)) ** 2)
    return 20.0 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10.0 * torch.log10(mse.clamp_min(1e-12))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def temporal_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean L1 error of consecutive intensity changes, [B,T,C,H,W]."""
    if pred.shape != target.shape or pred.ndim != 5 or pred.shape[1] < 2:
        raise ValueError("Temporal error requires matched [B,T,C,H,W] clips with T >= 2.")
    return ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs().mean()


def interpolation_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Framewise PSNR/MAE/SSIM on five unknown frames, temporal error on transitions."""
    from skimage.metrics import structural_similarity
    middle_pred, middle_gt = pred[:, 1:-1].float(), target[:, 1:-1].float()
    mse = (middle_pred - middle_gt).square().flatten(2).mean(-1)
    frame_psnr = -10.0 * mse.clamp_min(1e-12).log10()
    values = []
    for p, g in zip(middle_pred.flatten(0, 1).cpu().numpy(), middle_gt.flatten(0, 1).cpu().numpy()):
        size = min(7, p.shape[-2], p.shape[-1])
        if size % 2 == 0:
            size -= 1
        if size < 3:
            raise ValueError("SSIM needs images at least 3 x 3.")
        values.append(structural_similarity(g[0], p[0], data_range=1.0, win_size=size))
    return {"psnr": float(frame_psnr.mean()), "mae": float((middle_pred - middle_gt).abs().mean()),
            "ssim": float(sum(values) / len(values)), "temporal_error": float(temporal_error(pred, target))}


def compute_basic_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "psnr": float(psnr(pred, target).detach().cpu()),
        "mae": float(mae(pred, target).detach().cpu()),
    }
