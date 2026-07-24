from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
BASE_CONFIG = PROJECT_DIR / "configs" / "train_core.yaml"
DEFAULT_DATA_ROOT = os.environ.get("RAE_DATA_ROOT", str(PROJECT_DIR / "data" / "RLED"))
DEFAULT_GPU_IDS = "0,1"
DEFAULT_VARIANT = "full"
DEFAULT_SEED = 3407


VARIANTS = {
    "full",
    "supervised",
    "random_init",
    "wo_reliability",
    "wo_observability",
    "heuristic_gate",
    "wo_event_physics",
    "wo_quiet",
    "wo_temporal",
    "wo_reliability_loss",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def build_variant_config(
    variant: str,
    seed: int,
    data_root: str,
    stamp: str,
) -> Dict[str, Any]:
    config = copy.deepcopy(load_yaml(BASE_CONFIG))
    config["seed"] = int(seed)
    config["run_name"] = f"{variant}_seed{seed}_{stamp}"
    config["output_dir"] = str(PROJECT_DIR / "outputs" / "theory_v2" / "core")
    config["data"]["root"] = data_root

    if variant == "supervised":
        config["train"]["training_mode"] = "supervised_residual"
        config["loss"]["diffusion"] = 0.0
        config["loss"]["reconstruction"] = 1.0
    elif variant == "wo_reliability":
        config["model"]["use_reliability"] = False
        config["train"]["find_unused_parameters"] = True
        config["loss"]["reliability_calibration"] = 0.0
        config["loss"]["reliability_prior"] = 0.0
    elif variant == "wo_observability":
        config["model"]["use_observability"] = False
        config["train"]["find_unused_parameters"] = True
        config["loss"]["quiet"] = 0.0
    elif variant == "heuristic_gate":
        config["model"]["reliability_cue_channels"] = 0
        config["model"]["event_gate_source"] = "heuristic"
        config["theory"]["enabled"] = False
    elif variant == "wo_event_physics":
        config["loss"]["event_physics"] = 0.0
    elif variant == "wo_quiet":
        config["loss"]["quiet"] = 0.0
    elif variant == "wo_temporal":
        config["loss"]["temporal"] = 0.0
    elif variant == "wo_reliability_loss":
        config["loss"]["reliability_calibration"] = 0.0
        config["loss"]["reliability_prior"] = 0.0
    return config


def run_logged(command: list[str], env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Command:", " ".join(command), flush=True)
    print("Log:", log_path, flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(command)}")


def newest_checkpoint(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matches {pattern} under {root}.")
    return candidates[-1]


def save_random_init(config: Dict[str, Any], output_path: Path) -> None:
    import torch

    sys.path.insert(0, str(PROJECT_DIR / "src"))
    from raevdm.models import build_model

    model = build_model(config)
    payload = {
        "model": model.state_dict(),
        "model_ema": model.state_dict(),
        "epoch": -1,
        "step": 0,
        "config": config,
        "ablation": "random_init_residual_prior",
    }
    output_path.parent.mkdir(parents=True, exist_ok=False)
    torch.save(payload, output_path)


def run_geometry_pipeline(
    source_checkpoint: Path,
    variant: str,
    seed: int,
    stamp: str,
    env: Dict[str, str],
    log_dir: Path,
) -> Path:
    stages = [
        ("residual", "training/stages/train_residual.py", "best_middle.pt"),
        ("flow", "training/stages/train_flow_fusion.py", "best_flow_fusion.pt"),
        ("occlusion", "training/stages/train_occlusion_refinement.py", "best_occlusion_refinement.pt"),
        ("joint", "training/stages/train_joint.py", "FINAL_BEST_INFERENCE.pt"),
    ]
    current = source_checkpoint
    geometry_root = PROJECT_DIR / "outputs" / "theory_v2" / "stages"
    for stage, script, checkpoint_name in stages:
        prefix = f"theory_v2_{variant}_seed{seed}_{stage}_{stamp}"
        stage_env = dict(env)
        stage_env["RAE_RESUME_CHECKPOINT"] = str(current)
        stage_env["RAE_RUN_PREFIX"] = prefix
        stage_env["RAE_OUTPUT_DIR"] = str(geometry_root)
        before = set(geometry_root.glob(f"{prefix}_*")) if geometry_root.exists() else set()
        run_logged(
            [sys.executable, str(PROJECT_DIR / script)],
            stage_env,
            log_dir / f"{stage}.log",
        )
        after = set(geometry_root.glob(f"{prefix}_*"))
        created = sorted(after - before, key=lambda path: path.stat().st_mtime)
        if not created:
            raise RuntimeError(f"Stage {stage} completed but no unique run directory was found.")
        current = newest_checkpoint(created[-1], f"checkpoints/{checkpoint_name}")
        print(f"{stage} checkpoint: {current}", flush=True)
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Theory-aligned RaE-VDM training pipeline.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=DEFAULT_VARIANT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--gpus", default=DEFAULT_GPU_IDS)
    parser.add_argument("--stage", choices=("core", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not BASE_CONFIG.exists():
        raise FileNotFoundError(BASE_CONFIG)
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Training data root does not exist: {data_root}")
    raft_weights = Path(
        os.environ.get("RAE_RAFT_WEIGHTS", PROJECT_DIR / "raft_small_C_T_V2-01064c6d.pth")
    )
    if args.stage == "all" and not raft_weights.exists():
        raise FileNotFoundError(
            "RAFT weights are required for the geometry stages. Set "
            "RAE_RAFT_WEIGHTS or place raft_small_C_T_V2-01064c6d.pth "
            "in the repository root."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = build_variant_config(args.variant, args.seed, args.data_root, stamp)
    generated = PROJECT_DIR / "outputs" / "theory_v2" / "generated_configs"
    config_path = generated / f"{args.variant}_seed{args.seed}_{stamp}.yaml"
    save_yaml(config, config_path)

    log_dir = PROJECT_DIR / "logs" / "theory_v2" / f"{args.variant}_seed{args.seed}_{stamp}"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["RAE_GPU_IDS"] = args.gpus
    env["RAE_DATA_ROOT"] = args.data_root
    env["RAE_RAFT_WEIGHTS"] = str(raft_weights)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")

    print(f"Variant: {args.variant}")
    print(f"Seed: {args.seed}")
    print(f"Data: {args.data_root}")
    print(f"Config: {config_path}")
    print(f"Old checkpoints are never overwritten.")

    if args.variant == "random_init":
        random_path = (
            PROJECT_DIR
            / "outputs"
            / "theory_v2"
            / "core"
            / config["run_name"]
            / "checkpoints"
            / "random_init.pt"
        )
        save_random_init(config, random_path)
        core_checkpoint = random_path
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(args.gpus.split(','))}",
            "train.py",
            "--config",
            str(config_path),
        ]
        if args.smoke:
            command.append("--smoke")
        run_logged(command, env, log_dir / "core.log")
        run_dir = Path(config["output_dir"]) / config["run_name"]
        core_checkpoint = run_dir / "checkpoints" / "latest.pt"
        if not core_checkpoint.exists():
            raise FileNotFoundError(core_checkpoint)

    print(f"Core checkpoint: {core_checkpoint}", flush=True)
    if args.stage == "all" and not args.smoke:
        final_checkpoint = run_geometry_pipeline(
            core_checkpoint,
            args.variant,
            args.seed,
            stamp,
            env,
            log_dir,
        )
        print(f"FINAL checkpoint: {final_checkpoint}", flush=True)
    elif args.stage == "all" and args.smoke:
        print("Smoke mode stops after core validation; geometry stages are intentionally skipped.")


if __name__ == "__main__":
    main()
