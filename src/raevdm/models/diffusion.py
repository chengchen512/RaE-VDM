from __future__ import annotations

import torch
from torch import nn


def extract(values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, t)
    return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "linear",
    ) -> None:
        super().__init__()
        if schedule != "linear":
            raise ValueError("Only linear beta schedule is implemented in this prototype.")
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.timesteps = int(timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        ) * noise

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (xt - extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape) * eps) / extract(
            self.sqrt_alphas_cumprod, t, xt.shape
        ).clamp_min(1e-8)

    def ddim_timesteps(self, sampling_steps: int) -> torch.Tensor:
        sampling_steps = min(int(sampling_steps), self.timesteps)
        return torch.linspace(self.timesteps - 1, 0, sampling_steps).round().long()
