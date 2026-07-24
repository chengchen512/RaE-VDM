# Release Audit

This source release was assembled for public Git hosting.

## Included

- Core `raevdm` package and final model architecture.
- RLED data loading and event-voxel construction.
- Standard training, evaluation, and inference entry points.
- Residual, flow-fusion, occlusion-refinement, and joint training stages.
- Center-crop and tiled native-resolution inference.
- Generic YAML configurations, documentation, and dataset inspection tools.

## Excluded

- RaE-VDM and RAFT model weights.
- RLED/HQF data and all other datasets.
- Training logs, optimizer outputs, generated samples, and experiment archives.
- Paper tables, LaTeX sources, visualizations, and all figure-generation code.
- Virtual environments, IDE metadata, caches, and temporary files.
- Vendored SOTA repositories and their pretrained weights.

## Validation

- 30 Python files parsed successfully.
- Three focused YAML files and `pyproject.toml` parsed successfully.
- The release model loaded the final EMA checkpoint with `strict=True` and no
  missing or unexpected keys.
- Final architecture: 16,920,811 parameters and 289 checkpoint tensors.
- No plotting-library imports or `savefig` calls remain.
- No model/data/archive/media file extensions remain.
- No file in the release is larger than 1 MiB.
