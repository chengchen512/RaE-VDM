# Release Audit

This source release was updated from the supplied RaE-VDM.zip archive for
version 0.2.0. The README incorporates the current remote edits, and existing
assets/ demo files were retained.

## Included

- Core `raevdm` package and final model architecture.
- RLED data loading and event-voxel construction.
- Standard training, evaluation, and inference entry points.
- Four-stage training with end-to-end optimization in the final refinement stage.
- Shared deterministic inference and tiled native-resolution inference.
- Generic YAML configurations, documentation, and dataset inspection tools.
- Synthetic regression tests and an implementation-check utility.

## Excluded

- RaE-VDM and RAFT model weights.
- RLED/HQF data and all other datasets.
- Training logs, optimizer outputs, generated samples, and experiment archives.
- Paper tables, LaTeX sources, visualizations, and all figure-generation code.
- Virtual environments, IDE metadata, caches, and temporary files.
- Vendored SOTA repositories and their pretrained weights.

## Validation Provenance

`verification_report.json` is the report supplied with the archive. It records
28 passing tests and synthetic GPU checks, including frozen RAFT integration.
These are supplied results, not checks rerun while importing the archive. They
do not establish dataset accuracy or reproduce manuscript results.

Before removing the root training launchers and compatibility scripts, all
37 Python files passed syntax parsing, and
`pyproject.toml` and `verification_report.json` parsed successfully. Functional
tests were not rerun because the local Python environments lack PyTorch,
pytest, and the remaining project dependencies.

To validate the source in a configured environment, run:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python tools/check_implementation.py --device cpu
```

For a real RAFT integration check, pass local weights with `--raft-weights` and
use `--size 128` or larger. The retained demo assets and metrics predate this
source update. See [docs/migration.md](docs/migration.md) for changed checkpoint
semantics and implementation details.
