# Migration from the Supplied Source Archive

## Correctness Changes

- Restored src/raevdm/data, which was absent from the supplied archive even
  though all training/inference tools imported it. The original unanchored
  gitignore pattern data/ could also hide this Python package; it is now /data/.
- R^l and O^l condition the network. Count-domain S^h/Q^h independently define
  the residual gate. The paper path rejects learned/hybrid residual gates.
- V is used for physical targets and support mass, while U is used for the
  normalized neural condition. Physical targets no longer clip to [-2,2].
- The learned support is mu/(mu+kappa), not the mean reliability of normalized
  voxels. Supported confidence averages only bins with event activity.
- Both event/quiet loss weights stop gradients; the confidence-supervision
  target also stops gradients. R still receives event-target gradients.
- Refinement now exposes pred_raw before clipping/restoration. Endpoint loss
  can train the refinement head while final endpoints remain exact.
- Normalized target alpha and actual timestamps are passed to geometry. The
  central-time strength 4*alpha*(1-alpha) is computed separately for blending.
- Reconstruction training uses the deterministic branch and, when active, a
  second stochastic denoising call to the same predictor. It does not substitute
  a stochastic x0 estimate for the deployed residual output.
- Training is four stages totaling 170 epochs in the full recipe. The old
  separate train_joint.py has been removed; end-to-end optimization occupies
  the final three refinement epochs.
- Standard inference/evaluation now uses one deterministic predictor call and
  excludes GT intermediates from model input. Evaluation excludes anchors from
  PSNR/SSIM/MAE and averages results per sequence.
- Event voxelization honors half-open intervals and supports polarity-specific
  contrast thresholds. Tiled inference computes count percentiles on the full
  spatial field and restores endpoints after raised-cosine blending.

## Checkpoint Compatibility

Module/state names are retained, including legacy reliability_head names.
Shapes are checked strictly. A stage warm-start may introduce new flow or
refinement heads; mismatched core shapes and unexpected tensors are rejected.
No automatic zero-padding, truncation, or random substitution of old tensors
is performed.

The new preset has four anchor-cue channels. An older checkpoint with zero cue
channels can be instantiated from its saved architecture for evaluation. It
cannot initialize the four-cue preset without an explicitly designed migration
or new training. The runtime reads architecture dimensions from the saved
checkpoint and emits a legacy-semantics warning for pre-update checkpoints.

Tensor compatibility does not preserve numerical outputs: gating, physics,
time coordinates, and training objectives changed. Old weights therefore need
re-evaluation, and a claim that training produced the new objective requires
training/fine-tuning under the updated recipe. No such training was performed
while preparing this release.

## What Was Verified

See RELEASE_AUDIT.md for commands and verification scope. The checks use
synthetic event/frame fixtures, both analytic zero flow and locally available
frozen pretrained RAFT-Small on 128 x 128 synthetic anchors. They
exercise formulas, data reading, CLI inference/evaluation, every optimizer
phase, and GPU forward/backward passes. They do not establish dataset accuracy,
robustness improvements, or reproduction of the manuscript tables.

The original assets/ demo and its metrics are historical and are not new
measurements from this source update. No estimated or invented experiment
values were added.
