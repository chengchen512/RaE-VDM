from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

from .diffusion import DiffusionSchedule, extract
from .flow_fusion import FlowResidualFusionHead
from .occlusion_refinement import OcclusionAwareRefinementHead
from .reliability import ReliabilityCalibratedEventEncoder
from .unet3d import ConditionalVideoUNet
from ..event_statistics import count_priors


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


def linear_keyframe_interpolation(keyframes: torch.Tensor, keyframe_mask: torch.Tensor,
                                 timestamps: torch.Tensor | None = None) -> torch.Tensor:
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
            raise ValueError("At least two endpoint anchors are required.")
        if len(key_idx) == 1:
            out[b] = keyframes[b, key_idx[0]]
            continue
        first, last = key_idx[0], key_idx[-1]
        out[b, :first] = keyframes[b, first]
        out[b, last + 1 :] = keyframes[b, last]
        for left, right in zip(key_idx[:-1], key_idx[1:]):
            denom = max(right - left, 1)
            for frame_idx in range(left, right + 1):
                if timestamps is None:
                    weight = float(frame_idx - left) / float(denom)
                else:
                    time = timestamps[b] if timestamps.ndim == 2 else timestamps
                    weight = ((time[frame_idx] - time[left]) / (time[right] - time[left])).to(keyframes)
                out[b, frame_idx] = (1.0 - weight) * keyframes[b, left] + weight * keyframes[b, right]
    return out


class RaEVDM(nn.Module):
    """Event-use confidence, count-gated residual prediction, and geometry."""

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
        event_gate_source: str = "count",
        confidence_mode: str = "learned",
        confidence_threshold: float = 0.5,
        observability_mode: str = "learned",
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
        if self.event_gate_source not in {"count", "heuristic"}:
            raise ValueError("The current paper uses a fixed count-derived gate; choose event_gate_source=count.")
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
            confidence_mode=confidence_mode,
            confidence_threshold=confidence_threshold,
            observability_mode=observability_mode,
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
        linear = linear_keyframe_interpolation(batch["keyframes"].float(), batch["keyframe_mask"].float(), batch.get("timestamps"))
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
            linear_keyframe_interpolation(batch["keyframes"].float(), batch["keyframe_mask"].float(), batch.get("timestamps"))
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
        """Eqs. (23)--(26): the gate is independent of learned R and O."""
        if "event_counts" not in batch:
            raise ValueError("Count-derived gating requires unnormalized event_counts.")
        if not 0 <= gate_floor <= 1 or gate_power <= 0:
            raise ValueError("Invalid gate floor/power.")
        event_bins = _scalar_int(batch.get("event_bins", 5))
        with torch.no_grad():
            motion = batch.get("count_gate_motion")
            if motion is None:
                motion = count_priors(batch["event_counts"], event_bins)["count_motion"]
        if motion.shape[2] != frames_t - 1:
            raise ValueError("Count intervals do not match the requested frame sequence.")
        h, w = batch["keyframes"].shape[-2:]
        gate = motion.new_zeros(motion.shape[0], 1, frames_t, h, w)
        gate[:, :, 1:-1] = 0.5 * (motion[:, :, :-1] + motion[:, :, 1:])
        gate = F.avg_pool3d(gate, (1, 5, 5), stride=1, padding=(0, 2, 2))
        gate = gate_floor + (1.0 - gate_floor) * gate.clamp(0.0, 1.0).pow(gate_power)
        key_mask = video_to_bcthw(batch["keyframe_mask"].float())
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

        linear_btchw = linear_keyframe_interpolation(batch["keyframes"].float(), batch["keyframe_mask"].float(), batch.get("timestamps"))
        linear = video_to_bcthw(linear_btchw).clamp(0.0, 1.0)
        key_mask = video_to_bcthw(batch["keyframe_mask"].float()).clamp(0.0, 1.0)
        keyframes_int = video_to_bcthw(batch["keyframes"].float()).clamp(0.0, 1.0)
        keyframes = intensity_to_diffusion(keyframes_int) * key_mask
        bsz, _, frames_t, _, _ = linear.shape

        x = intensity_to_diffusion(linear)
        timestep = torch.full((bsz,), residual_timestep, device=linear.device, dtype=torch.long)
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
        pred_raw = linear + correction
        pred = pred_raw.clamp(0.0, 1.0)
        pred = pred * (1.0 - key_mask) + keyframes_int * key_mask
        return {
            "pred": pred,
            "pred_raw": pred_raw,
            "linear": linear,
            "correction": correction,
            "raw_residual": raw_residual,
            "event_gate": event_gate,
            "event": event_out,
            "gt": video_to_bcthw(batch["frames"].float()) if "frames" in batch else None,
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
        include_diffusion: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if include_diffusion:
            deterministic = self.forward(batch, mode=mode, flow_inputs=flow_inputs,
                                         residual_cfg=residual_cfg, base_raft_peak=base_raft_peak,
                                         detach_base=detach_base)
            return {**deterministic, "diffusion_outputs": self.forward(batch, t=t, mode="diffusion")}
        if mode == "joint_residual":
            residual = self.residual_correction(batch, residual_cfg)
            stochastic = self.forward(batch, t=t, mode="diffusion")
            return {**residual, "diffusion_outputs": stochastic}
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
                "x0_pred_raw": intensity_to_diffusion(residual_out["pred_raw"]),
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
    def reconstruct(self, batch: Dict[str, torch.Tensor], flow_inputs=None, residual_cfg=None,
                    base_raft_peak: float = 0.55) -> torch.Tensor:
        """One deterministic predictor call, with no ground-truth-frame input."""
        if self.use_occlusion_refinement:
            mode = "occlusion_refinement"
        elif self.use_flow_fusion:
            mode = "flow_fusion"
        else:
            mode = "supervised_residual"
        if self.use_flow_fusion:
            result = self(batch, mode=mode, flow_inputs=flow_inputs,
                          residual_cfg=residual_cfg, base_raft_peak=base_raft_peak)
        else:
            result = self.residual_correction(batch, residual_cfg)
        return result["pred"].permute(0, 2, 1, 3, 4).contiguous()

    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], sampling_steps: int = 50) -> torch.Tensor:
        """Legacy iterative DDIM diagnostic; use reconstruct for paper inference."""
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
