# Theory-to-Code Map

This file maps the intensity-video RaE-VDM theory to the implementation.

## Variables

| Theory term | Meaning | Code |
| --- | --- | --- |
| `S` | sparse intensity keyframes | `batch["keyframes"]` |
| `M` | keyframe mask | `batch["keyframe_mask"]` |
| `E` | degraded event stream | RLED `/events/xs,ys,ts,ps` |
| `V+`, `V-` | positive/negative event voxels | `batch["event_raw"]` |
| `R+`, `R-` | polarity reliability | `outputs["event"]["reliability"]` |
| `O` | no-event observability | `outputs["event"]["observability"]` |
| `X` | restored intensity video | `model.sample(...)` |
| `z_t` | noisy diffusion variable | `outputs["xt"]` |
| `epsilon_theta` | denoising network | `ConditionalVideoUNet` |

## Modules

| Paper module | Implementation |
| --- | --- |
| RLED event/image interface | `src/raevdm/data/rled_h5.py` |
| Reliability-calibrated representation | `src/raevdm/data/event_voxel.py`, `src/raevdm/models/reliability.py` |
| Event-conditioned video diffusion | `src/raevdm/models/raevdm.py` |
| Conditional 3D denoiser | `src/raevdm/models/unet3d.py` |
| DDPM/DDIM schedule | `src/raevdm/models/diffusion.py` |
| Training objective | `src/raevdm/losses.py` |

## Losses

| Loss | Code key | Purpose |
| --- | --- | --- |
| diffusion noise prediction | `diffusion` | Learns the video diffusion denoising prior. |
| event-physics consistency | `event_physics` | Matches predicted log-intensity change to reliability-weighted event evidence. |
| quiet-region regularization | `quiet` | Suppresses artificial motion in observable no-event regions. |
| keyframe anchoring | `anchor` | Keeps sparse observed frames fixed. |
| temporal regularity | `temporal` | Penalizes second-order temporal flicker. |
| reliability calibration | `reliability_calibration` | Aligns reliability with residual correctness. |
| reliability prior | `reliability_prior` | Stabilizes early training with a removable heuristic prior. |

## Main Ablations

- `model.use_reliability = false`
- `model.use_observability = false`
- set `loss.event_physics = 0`
- set `loss.quiet = 0`
- set `loss.anchor = 0`
- vary `data.keyframe_policy` and `data.keyframe_stride`
- vary `data.event_bins`
