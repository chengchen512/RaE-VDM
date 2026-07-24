# Final Training Stages

These files are not duplicate launchers. Each one optimizes a different part
of the final model and consumes the best checkpoint from the previous stage.

| Order | Script | Input | Output |
|---|---|---|---|
| 1 | `train_residual.py` | diffusion-core checkpoint | `best_middle.pt` |
| 2 | `train_flow_fusion.py` | `best_middle.pt` | `best_flow_fusion.pt` |
| 3 | `train_occlusion_refinement.py` | `best_flow_fusion.pt` | `best_occlusion_refinement.pt` |
| 4 | `train_joint.py` | `best_occlusion_refinement.pt` | `FINAL_BEST_INFERENCE.pt` |

Normally, run all stages through `python train_pipeline.py ...`. Execute an
individual stage only when resuming a partial pipeline; in that case set
`RAE_RESUME_CHECKPOINT` and the other environment variables documented in the
root README.

