#!/usr/bin/env python3
"""Ray Tune training orchestration for glacier mapping experiments."""

import argparse
import glob
import os
import subprocess
import sys
from typing import Dict, List, Any

import yaml

try:
    import ray
    from ray import tune

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not available. Install with: uv pip install ray[tune]")


def load_server_config(servers_yaml_path: str, server_name: str) -> Dict[str, Any]:
    """Load server configuration from servers.yaml."""
    with open(servers_yaml_path, "r") as f:
        servers = yaml.safe_load(f)

    if server_name not in servers:
        raise ValueError(f"Server '{server_name}' not found in {servers_yaml_path}")

    return servers[server_name]


def get_configs_for_server(
    server_name: str, patterns: List[str] | None = None
) -> List[str]:
    """Get all experiment configs assigned to specified server."""
    configs = []
    config_files = glob.glob("configs/experiments/*.yaml")

    for config_file in config_files:
        # Skip debug and initial validation configs unless explicitly requested
        if (
            any(x in config_file for x in ["debug_", "initial_validation"])
            and not patterns
        ):
            continue

        with open(config_file) as f:
            config = yaml.safe_load(f)
            if config.get("server") == server_name:
                # Apply pattern filtering if specified
                if patterns:
                    config_name = os.path.basename(config_file)
                    if not any(
                        pattern.replace("*", "") in config_name for pattern in patterns
                    ):
                        continue
                configs.append(config_file)

    return sorted(configs)


def get_gpu_resources(
    server_name: str, gpu_per_trial: float | None = None
) -> Dict[str, float]:
    """Calculate optimal GPU resources based on server hardware."""
    if gpu_per_trial is not None:
        return {"gpu": gpu_per_trial}

    resources = {
        "desktop": 0.5,  # 2 trials on 8GB RTX 3060 Ti
        "bilbo": 0.5,  # 2 trials on 24GB RTX 3090s (bigger datasets)
        "frodo": 0.33,  # 3 trials on 11GB RTX 2080 Tis
    }
    return {"gpu": resources[server_name]}


def ray_train(config: Dict[str, Any]):
    """Training function for Ray Tune."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train.py",
        "--config",
        config["config_path"],
        "--server",
        config["server"],
        "--max-epochs",
        str(config["max_epochs"]),
        "--mlflow-enabled",
        "true",
        "--tracking-uri",
        config["tracking_uri"],
        "--output-dir",
        config["output_path"],
    ]

    print(f"Starting training: {config['config_path']} on {config['server']}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Completed training: {config['config_path']}")
        return {"status": "completed", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        print(f"Failed training: {config['config_path']}, error: {e.stderr}")
        return {"status": "failed", "error": e.stderr}


def main():
    """Main orchestration function."""
    if not RAY_AVAILABLE:
        print("Error: Ray is not available. Install with: uv pip install ray[tune]")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Ray Tune training orchestration")
    parser.add_argument(
        "--server",
        required=True,
        choices=["desktop", "bilbo", "frodo"],
        help="Target server for experiment execution",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=500,
        help="Override max epochs for all experiments",
    )
    parser.add_argument(
        "--gpu-per-trial", type=float, help="Override GPU allocation per trial"
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        help="Run only configs matching these patterns (e.g., baseline_*)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous failed experiments"
    )
    parser.add_argument(
        "--tracking-uri",
        default="https://mlflow.developerjose.duckdns.org/",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per config (for hyperparameter sweeps)",
    )

    args = parser.parse_args()

    # Validate server and load configuration
    server_config = load_server_config("configs/servers.yaml", args.server)
    configs = get_configs_for_server(args.server, args.patterns)

    if not configs:
        print(f"No experiment configs found for server '{args.server}'")
        if args.patterns:
            print(f"with patterns: {args.patterns}")
        sys.exit(1)

    print(f"Found {len(configs)} experiment configs for server '{args.server}':")
    for config in configs:
        print(f"  - {config}")

    if args.dry_run:
        print("\nDry run - not executing experiments")
        return

    # Get GPU resources for this server
    gpu_resources = get_gpu_resources(args.server, args.gpu_per_trial)
    print(f"\nGPU allocation per trial: {gpu_resources}")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Prepare experiment configurations
    experiment_configs = []
    for config_path in configs:
        experiment_configs.append(
            {
                "config_path": config_path,
                "server": args.server,
                "max_epochs": args.max_epochs,
                "tracking_uri": args.tracking_uri,
                "output_path": server_config["output_path"],
            }
        )

    # Configure Ray Tune
    tuner = tune.Tuner(
        tune.with_parameters(ray_train),
        param_space={"config": tune.grid_search(experiment_configs)},
        run_config=tune.RunConfig(
            name=f"glacier_experiments_{args.server}",
            stop={"training_iteration": 1},  # Each config runs once
            failure_config=tune.FailureConfig(max_failures=3),
            verbose=1,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_score_attribute="status",
                checkpoint_score_order="max",
            ),
        ),
        tune_config=tune.TuneConfig(
            metric="status",
            mode="max",
            num_samples=args.num_samples,
        ),
    )

    print(f"\nStarting Ray Tune on server '{args.server}'...")
    print("Ray Tune dashboard: http://localhost:8265")
    print(f"MLflow tracking: {args.tracking_uri}")

    # Run experiments
    result_grid = tuner.fit()

    # Report results
    print(f"\nExperiment Results for {args.server}:")
    completed = 0
    failed = 0

    for result in result_grid:
        if result.metrics.get("status") == "completed":
            completed += 1
        else:
            failed += 1
            print(f"Failed: {result.config['config']['config_path']}")

    print(f"Completed: {completed}, Failed: {failed}")

    if failed > 0:
        print("\nTo resume failed experiments, run:")
        print(f"uv run python scripts/ray_train.py --server {args.server} --resume")


if __name__ == "__main__":
    main()
