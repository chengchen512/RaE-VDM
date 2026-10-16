# Revised Manuscript to Code

Cited paper: *Reliability-Aware Event-Guided Video Interpolation with Degraded
Event Streams*, supplied revised manuscript, Sec. III and Sec. IV-A.
Equation numbers below refer to that version.

## Symbols and Tensor Layout

Let B be batch size, T=K+1 frames, Be bins per frame interval, and N=(T-1)*Be.

| Paper | Tensor / implementation | Shape / units |
| --- | --- | --- |
| S, I0, IK (1) | batch.keyframes, batch.keyframe_mask | [B,T,1,H,W], intensity [0,1]; two endpoints |
| t_k, alpha_k (4) | batch.timestamps; linear_keyframe_interpolation | [B,T], timestamps; normalized target time |
| B, normalized B (4)-(5) | linear; intensity_to_diffusion | [B,1,T,H,W]; [0,1] / [-1,1] |
| V^{p} (12)-(13) | batch.event_counts | [B,2,N,H,W], unnormalized soft event counts, positive then negative |
| U^{p} (14) | batch.event_raw | [B,2,N,H,W], per-interval/per-polarity P99 normalization |
| R^{p,l} (10),(15) | event.reliability / event.event_use_confidence | [B,2,N,H,W], task-oriented event-use confidence |
| r_i^l (10) | event_confidence_at_events | Event-order scores using the original temporal interpolation weights |
| O^l, o_k^l (11) | event.observability; learned_event_statistics | [B,1,N,H,W], bin mean to [B,1,T-1,H,W] |
| D_R^{k,l} (16) | signed_event_delta(counts,R,Be,C) | [B,1,T-1,H,W], unnormalized log-intensity change; no [-2,2] clamp |
| mu^l, pi^l (17) | learned_mass, learned_support | Unsigned confidence mass; mu/(mu+kappa), kappa=1 |
| C_E (18) | event.condition | [B,7,N,H,W]: U+, U-, R+, R-, U+R+, U-R-, O |
| z_s, epsilon (19)-(21) | model.forward(mode="diffusion") | Stochastic training path of the shared 3D U-Net |
| Delta_theta (22) | residual_correction.raw_residual | Same U-Net, normalized interpolation base, fixed index 0 |
| S^{p,h}, Q^h (23)-(24) | count_priors.count_support/count_silence | Fixed statistics of V, not learned R/O |
| pi^h, q^h, omega^h (25) | count_precision/count_quiet/count_motion | Count-domain interval statistics |
| G (26) | RaEVDM.event_gate | Adjacent interval mean, 5x5 average, square root, floor 0.08; endpoints zero |
| X_res (27) | residual_correction.pred | B + 0.30*G*tanh(Delta/2), clipped and anchored |
| W0, WK, Cw, alpha (28) | geometry_inputs / synthesize_occlusion_aware_warp_video | Frozen RAFT-Small endpoint flows, forward splats, consistency and photometric agreement |
| Gamma, M_occ, M_hole, M_bd (29) | visibility0/1, occlusion_map, hole_map, motion_boundary | Geometry-derived diagnostics, no GT motion input |
| unconstrained X (29) | occlusion_refinement.pred_raw | Before clipping and endpoint restoration |
| final X (30) | pred; runtime.predict/predict_tiled | [B,T,1,H,W] at public API; range [0,1], exact anchors |

The legacy class/parameter identifiers containing "Reliability" are retained
to preserve checkpoint state-dictionary keys. R is an event-use confidence, not
a calibrated event-validity probability. Count support S^h and count silence Q^h
are not alternative names for learned R^l and O^l.

## Losses and Gradient Paths

Implementation: src/raevdm/losses.py; statistics: src/raevdm/event_statistics.py.

| Equation | Loss key | Essential implementation |
| --- | --- | --- |
| (21) | diffusion | MSE between injected Gaussian noise and shared-predictor estimate |
| (32) | reconstruction | L1 on the active stage's clipped, anchor-restored output |
| (33) | event_physics | sg(pi^l)*rho(D_pred-D_R); gradients retained through D_R |
| (34) | quiet | sg(o^l*(1-pi^l))*rho(relu(abs(D_pred)-0.02)) |
| (35) | anchor | L1 endpoint error of pred_raw, before clipping and projection |
| (36) | temporal | L1 second temporal difference of prediction, not GT temporal error |
| (37)-(38) | supported_confidence | Count-weighted R per bin, then mean over active bins only |
| (39)-(40) | reliability_calibration | Event-supported MSE to detached exp(-abs(D_Xsup-D_Eh)) |
| (40) | reliability_prior | Bin-level MSE(R,S^h) |
| (40) | reliability_supervision | 0.02*reliability_calibration + 0.01*reliability_prior |

Only reconstruction supervision at the residual stage and final refinement
activates the last row. Its outer coefficient is 1. The active reconstruction
is X_sup, and the entire confidence target is detached. Observability learns
through C_E; its quiet-loss weight is detached.

For stable batch/crop scaling, pixel sums in (32)-(34),(36) are implemented as
element means. Anchor error sums the means of the two endpoints. The support
normalization in (40) follows its explicitly stated denominator. Charbonnier
epsilon is 1e-3. These reduction choices are documented rather than silently
claiming a literal unnormalized sum at every crop size.

## Stage Schedule

configs/paper.yaml contains the complete schedule from Sec. IV-A:

- Pretraining: 120 epochs, diffusion weight 1, backbone LR 2e-4.
- Residual: 20 epochs, backbone LR 2e-5, diffusion weight 0.05, confidence weight 1.
- Flow: 3 frozen-backbone head epochs at 1e-4, then 12 joint epochs at 2e-6.
- Refinement: 3 head epochs; 9 refinement/fusion epochs; 3 end-to-end epochs.
  Head/fusion/backbone rates are 8e-5 / 8e-6 / 5e-7 when active.

Reconstruction / event / quiet / temporal weights are 1 / 0.05 / 0.01 / 0.005.
Anchor weight is 0.1 during refinement. Upstream diffusion/confidence objectives
are omitted during frozen phases. The core tensors and denoiser are shared
across stochastic and deterministic training; inference invokes the predictor
exactly once per clip or tile.

## Implementation Choices Beyond the Explicit Manuscript

The manuscript does not fully specify every architectural or data-organization
choice. The 48-channel encoder/backbone, four auxiliary anchor cues, optimizer
defaults, split holdout, fusion-head parametrization, and visibility cue formulas
are retained/adapted from the supplied implementation. They are executable
choices, not independently verified historical training provenance.

The main preset applies frame_step=3 to training as well as inference; the
manuscript explicitly states this stride for the main test protocol. The exact
historical training sampling manifest was not supplied and must be confirmed
before asserting an exact reproduction of published numbers.
