from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

from .diffusion import DiffusionSchedule, extract
from .flow_fusion import FlowResidualFusionHead
from .occlusion_refinement import OcclusionAwareRefinementHead
from .reliability import ReliabilityCalibratedEventEncoder
from .reliability import collapse_event_steps
from .unet3d import ConditionalVideoUNet


def video_to_bcthw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got {tuple(video.shape)}")
    return video.permute(0, 2, 1, 3, 4).contiguous()


def intensity_to_diffusion(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def diffusion_to_intensity(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def _scalar_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.flatten()[0].detach().cpu().item())
    return int(value)


def linear_keyframe_interpolation(keyframes: torch.Tensor, keyframe_mask: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation between available keyframes.

    Inputs and output use [B,T,C,H,W] in intensity range [0,1].
    """
    if keyframes.ndim != 5 or keyframe_mask.ndim != 5:
        raise ValueError("Expected keyframes and keyframe_mask with shape [B,T,C,H,W].")
    out = torch.zeros_like(keyframes)
    batch_size = keyframes.shape[0]
    for b in range(batch_size):
        frame_is_key = keyframe_mask[b].flatten(1).amax(dim=1) > 0.5
        key_idx = torch.nonzero(frame_is_key, as_tuple=False).flatten().tolist()
        if not key_idx:
            continue
        if len(key_idx) == 1:
            out[b] = keyframes[b, key_idx[0]]
            continue
        first, last = key_idx[0], key_idx[-1]
        out[b, :first] = keyframes[b, first]
        out[b, last + 1 :] = keyframes[b, last]
        for left, right in zip(key_idx[:-1], key_idx[1:]):
            denom = max(right - left, 1)
            for frame_idx in range(left, right + 1):
                weight = float(frame_idx - left) / float(denom)
                out[b, frame_idx] = (1.0 - weight) * keyframes[b, left] + weight * keyframes[b, right]
    return out


class RaEVDM(nn.Module):
    """Reliability-calibrated event-guided video diffusion for intensity clips."""

    def __init__(
        self,
        in_channels: int = 1,
        event_feature_channels: int = 32,
        base_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        time_embed_dim: int = 128,
        use_reliability: bool = True,
        use_observability: bool = True,
        reliability_cue_channels: int = 0,
        event_gate_source: str = "heuristic",
        use_linear_condition: bool = False,
        use_flow_fusion: bool = False,
        flow_fusion_hidden_channels: int = 32,
        flow_fusion_refinement_scale: float = 0.05,
        flow_fusion_alpha_delta_scale: float = 2.0,
        use_occlusion_refinement: bool = False,
        occlusion_refinement_hidden_channels: int = 32,
        occlusion_refinement_scale: float = 0.06,
        occlusion_branch_delta_scale: float = 0.30,
        diffusion_cfg: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        diffusion_cfg = diffusion_cfg or {}
        self.use_linear_condition = bool(use_linear_condition)
        self.reliability_cue_channels = int(max(reliability_cue_channels, 0))
        self.event_gate_source = str(event_gate_source).lower()
        if self.event_gate_source not in {"heuristic", "learned", "hybrid"}:
            raise ValueError("event_gate_source must be heuristic, learned, or hybrid.")
        self.use_flow_fusion = bool(use_flow_fusion)
        self.use_occlusion_refinement = bool(use_occlusion_refinement)
        if self.use_occlusion_refinement and not self.use_flow_fusion:
            raise ValueError("Occlusion refinement requires use_flow_fusion=True.")
        self.event_encoder = ReliabilityCalibratedEventEncoder(
            feature_channels=event_feature_channels,
            hidden_channels=event_feature_channels,
            use_reliability=use_reliability,
            use_observability=use_observability,
            cue_channels=self.reliability_cue_channels,
        )
        self.denoiser = ConditionalVideoUNet(
            in_channels=in_channels,
            condition_channels=event_feature_channels,
            extra_condition_channels=in_channels if self.use_linear_condition else 0,
            base_channels=base_channels,
            channel_mult=channel_mult,
            time_embed_dim=time_embed_dim,
        )
        self.schedule = DiffusionSchedule(**diffusion_cfg)
        self.flow_fusion = (
            FlowResidualFusionHead(
                hidden_channels=flow_fusion_hidden_channels,
                refinement_scale=flow_fusion_refinement_scale,
                alpha_delta_scale=flow_fusion_alpha_delta_scale,
            )
            if self.use_flow_fusion
            else None
        )
        self.occlusion_refinement = (
            OcclusionAwareRefinementHead(
                hidden_channels=occlusion_refinement_hidden_channels,
                refinement_scale=occlusion_refinement_scale,
                branch_delta_scale=occlusion_branch_delta_scale,
            )
            if self.use_occlusion_refinement
            else None
        )

    def linear_condition(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor | None:
        if not self.use_linear_condition:
            return None
        linear = linear_keyframe_interpolation(batch["keyframes"].float(), batch["keyframe_mask"].float())
        return intensity_to_diffusion(video_to_bcthw(linear))

    def encode_condition(self, batch: Dict[str, torch.Tensor], target_frames: int) -> Dict[str, torch.Tensor]:
        cues = self.reliability_cues(batch, target_frames)
        return self.event_encoder(
            event_raw=batch["event_raw"].float(),
            target_frames=target_frames,
            prior_reliability=batch.get("event_prior_reliability"),
            prior_observability=batch.get("event_prior_observability"),
            auxiliary_cues=cues,
        )

    def reliability_cues(
        self,
        batch: Dict[str, torch.Tensor],
        target_frames: int,
    ) -> torch.Tensor | None:
        """Build observation cues at the event-voxel temporal resolution.

        The four theory-aligned channels are interpolated intensity, darkness,
        saturation risk, and endpoint disagreement. Extra requested channels
        are zero-filled so configurations remain explicit and deterministic.
        """
        if self.reliability_cue_channels <= 0:
            return None
        linear = video_to_bcthw(
            linear_keyframe_interpolation(batch["keyframes"].float(), batch["keyframe_mask"].float())
        ).clamp(0.0, 1.0)
        interval_intensity = 0.5 * (linear[:, :, :-1] + linear[:, :, 1:])
        darkness = 1.0 - interval_intensity
        saturation_risk = (2.0 * torch.abs(interval_intensity - 0.5)).clamp(0.0, 1.0)
        endpoint_disagreement = torch.abs(linear[:, :, -1:] - linear[:, :, :1]).expand_as(interval_intensity)
        base = torch.cat([interval_intensity, darkness, saturation_risk, endpoint_disagreement], dim=1)
        if self.reliability_cue_channels < base.shape[1]:
            base = base[:, : self.reliability_cue_channels]
        elif self.reliability_cue_channels > base.shape[1]:
            padding = base.new_zeros(
                base.shape[0],
                self.reliability_cue_channels - base.shape[1],
                base.shape[2],
                base.shape[3],
                base.shape[4],
            )
            base = torch.cat([base, padding], dim=1)
        event_steps = batch["event_raw"].shape[2]
        return F.interpolate(
            base,
            size=(event_steps, base.shape[-2], base.shape[-1]),
            mode="trilinear",
            align_corners=False,
        )

    def event_gate(
        self,
        batch: Dict[str, torch.Tensor],
        frames_t: int,
        gate_floor: float = 0.08,
        gate_power: float = 0.50,
        event_out: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        device = batch["frames"].device
        h, w = batch["frames"].shape[-2:]
        learned_motion = None
        if self.event_gate_source in {"learned", "hybrid"} and event_out is not None:
            event_bins = _scalar_int(batch.get("event_bins", 5))
            raw = batch["event_raw"].float().clamp_min(0.0)
            reliability = event_out["reliability"].clamp(0.0, 1.0)
            rel_steps = (reliability * raw).sum(dim=1, keepdim=True) / raw.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            precision = collapse_event_steps(
                rel_steps, clip_len=frames_t, event_bins=event_bins, reduce="mean"
            ).clamp(0.0, 1.0)
            observability = collapse_event_steps(
                event_out["observability"], clip_len=frames_t, event_bins=event_bins, reduce="mean"
            ).clamp(0.0, 1.0)
            learned_motion = (1.0 - observability) * (0.25 + 0.75 * precision)

        if "event_interval_observability" in batch:
            obs = video_to_bcthw(batch["event_interval_observability"].float()).clamp(0.0, 1.0)
            motion = (1.0 - obs).clamp(0.0, 1.0)
            if "event_interval_precision" in batch:
                precision = video_to_bcthw(batch["event_interval_precision"].float()).clamp(0.0, 1.0)
                motion = motion * (0.25 + 0.75 * precision)
        else:
            raw = batch["event_raw"].float().abs().sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            event_bins = batch.get("event_bins", 5)
            if torch.is_tensor(event_bins):
                event_bins = int(event_bins.flatten()[0].detach().cpu().item())
            motion = collapse_event_steps(raw, clip_len=frames_t, event_bins=int(event_bins), reduce="mean").clamp(0.0, 1.0)

        if self.event_gate_source == "learned" and learned_motion is not None:
            motion = learned_motion
        elif self.event_gate_source == "hybrid" and learned_motion is not None:
            motion = 0.5 * motion + 0.5 * learned_motion

        gate = torch.zeros((motion.shape[0], 1, frames_t, h, w), device=device, dtype=motion.dtype)
        intervals = motion.shape[2]
        for frame_idx in range(frames_t):
            if frame_idx == 0:
                value = motion[:, :, 0]
            elif frame_idx >= frames_t - 1:
                value = motion[:, :, -1]
            else:
                left = min(frame_idx - 1, intervals - 1)
                right = min(frame_idx, intervals - 1)
                value = 0.5 * (motion[:, :, left] + motion[:, :, right])
            gate[:, :, frame_idx] = value
        gate = F.avg_pool3d(gate, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
        gate = gate.clamp(0.0, 1.0).pow(float(gate_power))
        gate = (float(gate_floor) + (1.0 - float(gate_floor)) * gate).clamp(0.0, 1.0)
        key_mask = video_to_bcthw(batch["keyframe_mask"].float()).clamp(0.0, 1.0)
        return gate * (1.0 - key_mask)

    def residual_correction(
        self,
        batch: Dict[str, torch.Tensor],
        residual_cfg: Dict[str, Any] | None = None,
    ) -> Dict[str, torch.Tensor]:
        residual_cfg = residual_cfg or {}
        residual_scale = float(residual_cfg.get("residual_scale", 0.30))
        residual_timestep = int(residual_cfg.get("residual_timestep", 0))
        temperature = float(residual_cfg.get("residual_tanh_temperature", 2.0))
        gate_floor = float(residual_cfg.get("event_gate_floor", 0.08))
        gate_power = float(residual_cfg.get("event_gate_power", 0.50))

        gt = video_to_bcthw(batch["frames"].float())
        linear_btchw = linear_keyframe_interpolation(batch["keyframes"].float(), batch["keyframe_mask"].float())
        linear = video_to_bcthw(linear_btchw).clamp(0.0, 1.0)
        key_mask = video_to_bcthw(batch["keyframe_mask"].float()).clamp(0.0, 1.0)
        keyframes_int = video_to_bcthw(batch["keyframes"].float()).clamp(0.0, 1.0)
        keyframes = intensity_to_diffusion(keyframes_int) * key_mask
        bsz, _, frames_t, _, _ = gt.shape

        x = intensity_to_diffusion(linear)
        timestep = torch.full((bsz,), residual_timestep, device=gt.device, dtype=torch.long)
        event_out = self.encode_condition(batch, frames_t)
        extra_condition = self.linear_condition(batch)
        raw_residual = self.denoiser(
            x,
            timestep,
            event_out["features"],
            keyframes,
            key_mask,
            extra_condition=extra_condition,
        )
        event_gate = self.event_gate(
            batch,
            frames_t,
            gate_floor=gate_floor,
            gate_power=gate_power,
            event_out=event_out,
        )
        correction = residual_scale * torch.tanh(raw_residual / temperature) * event_gate
        correction = correction * (1.0 - key_mask)
        pred = (linear + correction).clamp(0.0, 1.0)
        pred = pred * (1.0 - key_mask) + keyframes_int * key_mask
        return {
            "pred": pred,
            "linear": linear,
            "correction": correction,
            "raw_residual": raw_residual,
            "event_gate": event_gate,
            "event": event_out,
            "gt": gt,
            "keyframe_mask": key_mask,
        }

    def flow_fusion_correction(
        self,
        batch: Dict[str, torch.Tensor],
        flow_inputs: Dict[str, torch.Tensor],
        residual_cfg: Dict[str, Any] | None = None,
        base_raft_peak: float = 0.55,
        detach_base: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if self.flow_fusion is None:
            raise RuntimeError("Flow fusion is disabled. Set model.use_flow_fusion=true.")
        if detach_base:
            with torch.no_grad():
                residual_out = self.residual_correction(batch, residual_cfg)
        else:
            residual_out = self.residual_correction(batch, residual_cfg)

        fused = self.flow_fusion(
            residual_video=residual_out["pred"],
            warped_video=flow_inputs["warped_video"],
            linear_video=residual_out["linear"],
            event_gate=residual_out["event_gate"],
            warp_confidence=flow_inputs["warp_confidence"],
            time_map=flow_inputs["time_map"],
            keyframe_mask=residual_out["keyframe_mask"],
            base_raft_peak=base_raft_peak,
        )
        return {
            **residual_out,
            "residual_pred": residual_out["pred"],
            **fused,
            **flow_inputs,
        }

    def occlusion_refinement_correction(
        self,
        batch: Dict[str, torch.Tensor],
        flow_inputs: Dict[str, torch.Tensor],
        residual_cfg: Dict[str, Any] | None = None,
        base_raft_peak: float = 0.55,
        detach_base: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if self.occlusion_refinement is None:
            raise RuntimeError(
                "Occlusion refinement is disabled. "
                "Set model.use_occlusion_refinement=true."
            )
        required = (
            "endpoint_warp0",
            "endpoint_warp1",
            "visibility0",
            "visibility1",
            "occlusion_map",
            "hole_map",
            "motion_boundary",
        )
        missing = [key for key in required if key not in flow_inputs]
        if missing:
            raise ValueError(f"Missing occlusion-aware flow inputs: {missing}")

        flow_out = self.flow_fusion_correction(
            batch,
            flow_inputs,
            residual_cfg=residual_cfg,
            base_raft_peak=base_raft_peak,
            detach_base=detach_base,
        )
        base_prediction = flow_out["pred"]
        refined = self.occlusion_refinement(
            base_prediction=base_prediction,
            residual_video=flow_out["residual_pred"],
            warped_video=flow_out["warped_video"],
            linear_video=flow_out["linear"],
            endpoint_warp0=flow_out["endpoint_warp0"],
            endpoint_warp1=flow_out["endpoint_warp1"],
            event_gate=flow_out["event_gate"],
            warp_confidence=flow_out["warp_confidence"],
            visibility0=flow_out["visibility0"],
            visibility1=flow_out["visibility1"],
            occlusion_map=flow_out["occlusion_map"],
            hole_map=flow_out["hole_map"],
            motion_boundary=flow_out["motion_boundary"],
            time_map=flow_out["time_map"],
            keyframe_mask=flow_out["keyframe_mask"],
        )
        return {
            **flow_out,
            "flow_fusion_pred": base_prediction,
            **refined,
        }

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        t: torch.Tensor | None = None,
        mode: str = "diffusion",
        flow_inputs: Dict[str, torch.Tensor] | None = None,
        residual_cfg: Dict[str, Any] | None = None,
        base_raft_peak: float = 0.55,
        detach_base: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if mode == "flow_fusion":
            if flow_inputs is None:
                raise ValueError("flow_inputs are required for flow_fusion mode.")
            return self.flow_fusion_correction(
                batch,
                flow_inputs,
                residual_cfg=residual_cfg,
                base_raft_peak=base_raft_peak,
                detach_base=detach_base,
            )
        if mode == "occlusion_refinement":
            if flow_inputs is None:
                raise ValueError("flow_inputs are required for occlusion_refinement mode.")
            return self.occlusion_refinement_correction(
                batch,
                flow_inputs,
                residual_cfg=residual_cfg,
                base_raft_peak=base_raft_peak,
                detach_base=detach_base,
            )
        if mode == "supervised_residual":
            residual_out = self.residual_correction(batch, residual_cfg=residual_cfg)
            x0_pred = intensity_to_diffusion(residual_out["pred"])
            x0 = intensity_to_diffusion(residual_out["gt"])
            zeros = torch.zeros_like(x0)
            return {
                "t": torch.zeros(x0.shape[0], device=x0.device, dtype=torch.long),
                "noise": zeros,
                "xt": intensity_to_diffusion(residual_out["linear"]),
                "eps_pred": zeros,
                "x0_pred_raw": x0_pred,
                "x0_pred": x0_pred,
                "x0": x0,
                "event": residual_out["event"],
                "residual": residual_out,
            }
        if mode != "diffusion":
            raise ValueError(f"Unknown forward mode: {mode}")

        frames = video_to_bcthw(batch["frames"].float())
        x0 = intensity_to_diffusion(frames)
        b, _, frames_t, _, _ = x0.shape
        if t is None:
            t = self.schedule.sample_timesteps(b, x0.device)
        noise = torch.randn_like(x0)
        xt = self.schedule.q_sample(x0, t, noise)

        key_mask = video_to_bcthw(batch["keyframe_mask"].float())
        keyframes = intensity_to_diffusion(video_to_bcthw(batch["keyframes"].float())) * key_mask
        event_out = self.encode_condition(batch, frames_t)
        extra_condition = self.linear_condition(batch)
        eps_pred = self.denoiser(xt, t, event_out["features"], keyframes, key_mask, extra_condition=extra_condition)
        x0_pred_raw = self.schedule.predict_x0_from_eps(xt, t, eps_pred)
        x0_pred = x0_pred_raw * (1.0 - key_mask) + intensity_to_diffusion(video_to_bcthw(batch["keyframes"].float())) * key_mask

        return {
            "t": t,
            "noise": noise,
            "xt": xt,
            "eps_pred": eps_pred,
            "x0_pred_raw": x0_pred_raw,
            "x0_pred": x0_pred,
            "x0": x0,
            "event": event_out,
        }

    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], sampling_steps: int = 50) -> torch.Tensor:
        frames = video_to_bcthw(batch["frames"].float())
        b, c, frames_t, h, w = frames.shape
        device = frames.device
        key_mask = video_to_bcthw(batch["keyframe_mask"].float())
        keyframes = intensity_to_diffusion(video_to_bcthw(batch["keyframes"].float())) * key_mask
        event_out = self.encode_condition(batch, frames_t)
        extra_condition = self.linear_condition(batch)

        x = torch.randn((b, c, frames_t, h, w), device=device)
        x = x * (1.0 - key_mask) + keyframes * key_mask

        timesteps = self.schedule.ddim_timesteps(sampling_steps).to(device)
        for i, t_scalar in enumerate(timesteps):
            t = torch.full((b,), int(t_scalar.item()), device=device, dtype=torch.long)
            eps = self.denoiser(x, t, event_out["features"], keyframes, key_mask, extra_condition=extra_condition)
            x0 = self.schedule.predict_x0_from_eps(x, t, eps).clamp(-1.0, 1.0)
            x0 = x0 * (1.0 - key_mask) + keyframes * key_mask
            if i == len(timesteps) - 1:
                x = x0
            else:
                t_prev_scalar = timesteps[i + 1]
                alpha_prev = self.schedule.alphas_cumprod[int(t_prev_scalar.item())].to(device)
                x = torch.sqrt(alpha_prev) * x0 + torch.sqrt(1.0 - alpha_prev) * eps
                x = x * (1.0 - key_mask) + keyframes * key_mask
        return diffusion_to_intensity(x).permute(0, 2, 1, 3, 4).contiguous()


def build_model(config: Dict[str, Any]) -> RaEVDM:
    model_cfg = dict(config.get("model", {}))
    diffusion_cfg = dict(config.get("diffusion", {}))
    diffusion_cfg.pop("sampling_steps", None)
    if "channel_mult" in model_cfg:
        model_cfg["channel_mult"] = tuple(model_cfg["channel_mult"])
    return RaEVDM(**model_cfg, diffusion_cfg=diffusion_cfg)
