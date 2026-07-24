from __future__ import annotations

import math
from typing import Dict

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred.clamp(0, max_val) - target.clamp(0, max_val)) ** 2)
    return 20.0 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10.0 * torch.log10(mse.clamp_min(1e-12))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def compute_basic_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "psnr": float(psnr(pred, target).detach().cpu()),
        "mae": float(mae(pred, target).detach().cpu()),
    }
