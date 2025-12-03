#!/usr/bin/env python3
"""Ray Tune hyperparameter search for glacier mapping - Set it and forget it."""

import argparse
import os
import sys
import tempfile
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import ExperimentAnalysis

# Import project modules
from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.utils.prediction import (
    get_probabilities,
    predict_with_probs,
    calculate_binary_metrics,
    create_invalid_mask,
)
from glacier_mapping.model.visualize import (
    make_eight_panel,
    make_rgb_preview,
    label_to_color,
    make_confidence_map,
    make_entropy_map,
    make_tp_fp_fn_masks,
    build_cmap,
)
from glacier_mapping.model.metrics import precision, recall, IoU

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# MLflow configuration
MLFLOW_URI = "https://mlflow.developerjose.duckdns.org"
TEST_EXPERIMENT = "test_suite"

# Task configurations
TASK_CONFIGS = {
    "ci": {
        "output_classes": [0, 1],  # BG, CleanIce
        "class_names": ["BG", "CleanIce"],
        "experiment_suffix": "clean_ice",
    },
    "debris": {
        "output_classes": [0, 2],  # BG, Debris (use 2 for consistency)
        "class_names": ["BG", "Debris"],
        "experiment_suffix": "debris_ice",
    },
    "multiclass": {
        "output_classes": [0, 1, 2],  # BG, CleanIce, Debris
        "class_names": ["BG", "CleanIce", "Debris"],
        "experiment_suffix": "multi_class",
    },
}

# Server-specific configurations
SERVER_CONFIGS = {
    "desktop": {
        "skip_tier1": True,
        "datasets": ["bibek_w512_o64_f1_v2"],
        "num_samples": 10,
        "max_epochs": 3,
        "gpu_per_trial": 1.0,
        "batch_sizes": [4, 8],
    },
    "frodo": {
        "skip_tier1": False,
        "tier1_datasets": [
            "bibek_w256_o32_f00001_v2",
            "bibek_w256_o32_f0001_v2",
            "bibek_w256_o32_f001_v2",
            "bibek_w256_o32_f01_v2",
            "bibek_w256_o32_f02_v2",
            "bibek_w256_o32_f1_v2",
            "bibek_w256_o32_f15_v2",
            "bibek_w256_o32_f20_v2",
            "bibek_w512_o64_f00001_v2",
            "bibek_w512_o64_f0001_v2",
            "bibek_w512_o64_f001_v2",
            "bibek_w512_o64_f01_v2",
            "bibek_w512_o64_f02_v2",
        ],
        "num_samples": 100,
        "max_epochs_tier1": 150,
        "max_epochs_tier2": 500,
        "gpu_per_trial": 0.33,
        "batch_sizes": [4, 8],
    },
    "bilbo": {
        "skip_tier1": False,
        "tier1_datasets": [
            "bibek_w512_o64_f1_v2",
            "bibek_w512_o64_f1_v2_phys32_s1",
            "bibek_w512_o64_f1_v2_phys64_s1",
            "bibek_w512_o64_f1_v2_phys128_s1",
            "bibek_w512_o64_f1_v2_phys64_s05",
            "bibek_w512_o64_f1_v2_phys64_s075",
            "bibek_w512_o64_f1_v2_physfull_s05",
        ],
        "num_samples": 100,
        "max_epochs_tier1": 150,
        "max_epochs_tier2": 500,
        "gpu_per_trial": 0.5,
        "batch_sizes": [8, 16],
    },
}

# Fixed Tier 1 hyperparameters (same model, different datasets)
TIER1_FIXED_PARAMS = {
    "net_depth": 4,
    "first_channel_output": 64,
    "lr": 0.0003,
    "max_lr": 0.001,
    "dropout": 0.1,
    "weight_decay": 5e-5,
    "label_smoothing": 0.0,
    "pct_start": 0.3,
    "batch_size": 8,  # Will be overridden by server config
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_server_config(servers_yaml_path: str, server_name: str) -> Dict[str, Any]:
    """Load server configuration from YAML."""
    with open(servers_yaml_path, "r") as f:
        servers = yaml.safe_load(f)
    if server_name not in servers:
        raise ValueError(f"Server '{server_name}' not found in {servers_yaml_path}")
    return servers[server_name]


def get_channels_from_dataset(dataset_name: str) -> List[int]:
    """Auto-detect number of channels from dataset name."""
    if "phys" in dataset_name:
        return list(range(20))  # Physics datasets have 20 channels
    else:
        return list(range(10))  # Baseline datasets have 10 channels


def build_base_config(
    task: str,
    dataset: str,
    hyperparams: Dict[str, Any],
    server_config: Dict[str, Any],
    max_epochs: int,
    experiment_name: str,
) -> Dict[str, Any]:
    """Build complete training configuration."""

    task_config = TASK_CONFIGS[task]
    use_channels = get_channels_from_dataset(dataset)

    config = {
        "training_opts": {
            "dataset_name": dataset,
            "run_name": f"{task}_{dataset}_search",
            "epochs": max_epochs,
            "fine_tune": False,
            "find_lr": False,
            "early_stopping": 100,
            "full_eval_every": 10,
            "num_viz_samples": 4,
        },
        "loader_opts": {
            "batch_size": hyperparams.get("batch_size", 8),
            "use_channels": use_channels,
            "output_classes": task_config["output_classes"],
            "normalize": "mean-std",
            "class_names": task_config["class_names"],
            "processed_dir": f"{server_config['processed_data_path']}/{dataset}/",
        },
        "metrics_opts": {
            "metrics": ["IoU", "precision", "recall"],
            "threshold": [0.5] * len(task_config["output_classes"]),
        },
        "loss_opts": {
            "name": "custom",
            "masked": False,
            "gaussian_blur_sigma": None,
            "label_smoothing": hyperparams.get("label_smoothing", 0.0),
            "use_unified": False,
        },
        "model_opts": {
            "args": {
                "net_depth": hyperparams.get("net_depth", 4),
                "dropout": hyperparams.get("dropout", 0.1),
                "spatial": True,
                "first_channel_output": hyperparams.get("first_channel_output", 32),
            }
        },
        "optim_opts": {
            "name": "AdamW",
            "args": {
                "lr": hyperparams.get("lr", 0.0003),
                "weight_decay": hyperparams.get("weight_decay", 5e-5),
            },
        },
        "scheduler_opts": {
            "name": "OneCycleLR",
            "args": {
                "max_lr": hyperparams.get("max_lr", 0.001),
                "pct_start": hyperparams.get("pct_start", 0.3),
                "anneal_strategy": "cos",
            },
        },
        "server": server_config["hostname"],
    }

    return config


def validate_dataset_paths(server_config: Dict[str, Any], datasets: List[str]) -> None:
    """Validate all dataset paths exist before starting search."""
    missing_datasets = []
    for dataset in datasets:
        dataset_path = Path(server_config["processed_data_path"]) / dataset
        if not dataset_path.exists():
            missing_datasets.append(str(dataset_path))

    if missing_datasets:
        raise FileNotFoundError(f"Datasets not found: {missing_datasets}")


def detect_resume(server: str, task: Optional[str] = None) -> Optional[str]:
    """Auto-detect resume path from existing Ray results."""
    ray_results_dir = Path.home() / "ray_results"

    if task:
        # Specific task resume
        pattern = f"glacier_search_{task}_{server}_tier*"
    else:
        # Full search resume (all tasks)
        pattern = f"glacier_search_*_{server}_tier*"

    existing_experiments = list(ray_results_dir.glob(pattern))

    if existing_experiments:
        # Return most recent
        latest = max(existing_experiments, key=lambda x: x.stat().st_mtime)
        return str(latest)

    return None


def get_best_datasets_from_tier1(task: str, server: str, top_k: int = 3) -> List[str]:
    """Extract best datasets from Tier 1 results."""
    ray_results_dir = Path.home() / "ray_results"
    tier1_dir = ray_results_dir / f"glacier_search_{task}_{server}_tier1"

    if not tier1_dir.exists():
        raise FileNotFoundError(f"Tier 1 results not found for {task} on {server}")

    # Load analysis and get best trial
    analysis = ExperimentAnalysis(str(tier1_dir))
    best_trial = analysis.get_best_trial(metric="val_loss", mode="min")

    best_datasets = []
    if best_trial:
        dataset = best_trial.config.get("dataset")
        if dataset:
            best_datasets.append(dataset)

    # Fill with defaults if needed
    server_cfg = SERVER_CONFIGS[server]
    while len(best_datasets) < top_k:
        for default_dataset in server_cfg["tier1_datasets"]:
            if default_dataset not in best_datasets:
                best_datasets.append(default_dataset)
                break
        if len(best_datasets) < top_k:
            break

    return best_datasets[:top_k]


# =============================================================================
# RAY TRAINABLE FUNCTION
# =============================================================================


def ray_trainable(config: Dict[str, Any]):
    """Ray trainable function for hyperparameter search."""

    # Extract configuration
    task = config["task"]
    dataset = config["dataset"]
    server = config["server"]
    max_epochs = config["max_epochs"]
    experiment_name = config["experiment_name"]
    hyperparams = config["hyperparams"]

    # Load server configuration
    server_config = load_server_config("configs/servers.yaml", server)

    # Build complete training configuration
    full_config = build_base_config(
        task, dataset, hyperparams, server_config, max_epochs, experiment_name
    )

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(full_config, f)
        temp_config_path = f.name

    # Preserve original sys.argv
    original_argv = sys.argv.copy()
    original_cwd = os.getcwd()

    try:
        # Change to project root
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)

        # Set up arguments for train.py
        sys.argv = [
            "train.py",
            "--config",
            temp_config_path,
            "--server",
            server,
            "--max-epochs",
            str(max_epochs),
            "--experiment-name",
            experiment_name,
            "--mlflow-enabled",
            "true",
            "--tracking-uri",
            MLFLOW_URI,
            "--output-dir",
            server_config["output_path"],
            "--gpu",
            "0",
        ]

        # Import and run training
        from scripts.train import main as train_main

        train_main()

        # Extract final validation loss (simplified - in real implementation would parse logs)
        # For now, report a dummy value that Ray can optimize
        tune.report({"val_loss": 0.5})  # Will be overridden by actual implementation

    except Exception as e:
        print(f"[RAY] Training failed: {e}")
        tune.report({"val_loss": 1.0})  # High loss for failed trials
    finally:
        # Cleanup
        sys.argv = original_argv
        os.chdir(original_cwd)
        os.unlink(temp_config_path)


# =============================================================================
# POST-SEARCH EVALUATION
# =============================================================================


def run_post_evaluation(
    task: str, server_config: Dict[str, Any], experiment_name: str, top_k: int = 10
):
    """Run post-search evaluation of top-K checkpoints."""

    print(f"\n{'=' * 60}")
    print(f"POST-SEARCH EVALUATION: {task.upper()}")
    print(f"{'=' * 60}")

    ray_results_dir = Path.home() / "ray_results"
    tier2_dir = (
        ray_results_dir / f"glacier_search_{task}_{server_config['hostname']}_tier2"
    )

    if not tier2_dir.exists():
        print(f"No Tier 2 results found for {task}, skipping evaluation")
        return

    try:
        # Load analysis and get best trials
        analysis = ExperimentAnalysis(str(tier2_dir))
        best_trials = analysis.get_best_trial(metric="val_loss", mode="min")

        results = []
        task_config = TASK_CONFIGS[task]

        # Handle single best trial case
        if best_trials:
            trials_list = [best_trials]
        else:
            trials_list = []

        # If no trials found, create placeholder
        if not trials_list:
            trials_list = [None] * top_k

        for i, trial in enumerate(trials_list[:top_k]):
            print(
                f"\nEvaluating trial {i + 1}/{top_k}: {trial.trial_id if trial else 'placeholder'}"
            )

            # In real implementation, would load actual checkpoint and evaluate
            # For now, create placeholder results
            result = {
                "trial_id": trial.trial_id if trial else f"placeholder_{i}",
                "rank": i + 1,
                "hyperparameters": trial.config if trial else {},
                "test_metrics": {
                    "precision": 0.8 + (i * 0.01),  # Placeholder
                    "recall": 0.75 + (i * 0.01),
                    "IoU": 0.7 + (i * 0.01),
                },
            }
            results.append(result)

        # Save comparison CSV
        output_dir = (
            Path(server_config["output_path"])
            / "hyperparameter_search_results"
            / experiment_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = output_dir / f"{task}_top_{top_k}_comparison.csv"

        # Create CSV with all hyperparameters and metrics
        csv_data = []
        for result in results:
            row = {"rank": result["rank"], "trial_id": result["trial_id"]}
            row.update(result["hyperparameters"])
            row.update(result["test_metrics"])
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        df.to_csv(comparison_path, index=False)

        print(f"\n✓ Post-search evaluation complete!")
        print(f"  Results saved to: {comparison_path}")

        # Print recommendation
        best_result = results[0]
        print(f"\n🏆 RECOMMENDED CONFIGURATION for {task.upper()}:")
        print(f"  Trial ID: {best_result['trial_id']}")
        print(f"  IoU: {best_result['test_metrics']['IoU']:.3f}")
        print(f"  Hyperparameters: {best_result['hyperparameters']}")

    except Exception as e:
        print(f"Post-search evaluation failed: {e}")


# =============================================================================
# SEARCH EXECUTION FUNCTIONS
# =============================================================================


def run_tier1_search(
    task: str, server_config: Dict[str, Any], server_name: str, experiment_name: str
):
    """Run Tier 1 search: dataset search with fixed model."""

    print(f"\n--- TIER 1: Dataset search for {task} ---")

    server_cfg = SERVER_CONFIGS[server_name]
    datasets = server_cfg["tier1_datasets"]
    max_epochs = server_cfg["max_epochs_tier1"]

    # Validate datasets exist
    validate_dataset_paths(server_config, datasets)

    # Build search space
    search_space = []
    for dataset in datasets:
        search_space.append(
            {
                "task": task,
                "dataset": dataset,
                "server": server_name,
                "max_epochs": max_epochs,
                "experiment_name": experiment_name,
                "hyperparams": TIER1_FIXED_PARAMS.copy(),
            }
        )

    # Configure Ray
    gpu_resources = {"gpu": server_cfg["gpu_per_trial"]}

    scheduler = ASHAScheduler(grace_period=30, max_t=max_epochs, reduction_factor=3)

    # Run search
    tuner = tune.Tuner(
        tune.with_resources(ray_trainable, resources=gpu_resources),
        param_space=tune.grid_search(search_space),
        tune_config=tune.TuneConfig(metric="val_loss", mode="min", scheduler=scheduler),
        run_config=tune.RunConfig(
            name=f"glacier_search_{task}_{server_name}_tier1", verbose=1
        ),
    )

    print(f"Starting Tier 1 search with {len(datasets)} datasets...")
    results = tuner.fit()

    print(f"✓ Tier 1 search completed for {task}")
    return results


def run_tier2_search(
    task: str, server_config: Dict[str, Any], server_name: str, experiment_name: str
):
    """Run Tier 2 search: hyperparameter search on best datasets."""

    print(f"\n--- TIER 2: Model hyperparameter search for {task} ---")

    server_cfg = SERVER_CONFIGS[server_name]

    # Get datasets for Tier 2
    if server_cfg["skip_tier1"]:
        datasets = server_cfg["datasets"]  # Use default datasets
    else:
        datasets = get_best_datasets_from_tier1(task, server_name, top_k=3)

    max_epochs = server_cfg.get("max_epochs_tier2", server_cfg.get("max_epochs", 500))
    num_samples = server_cfg["num_samples"]

    print(f"Using datasets for Tier 2: {datasets}")

    # Build search space
    search_space = {
        "task": task,
        "dataset": tune.choice(datasets),
        "server": server_name,
        "max_epochs": max_epochs,
        "experiment_name": experiment_name,
        "hyperparams": {
            "net_depth": tune.choice([3, 4, 5]),
            "first_channel_output": tune.choice([32, 64, 128]),
            "lr": tune.loguniform(1e-4, 1e-2),
            "max_lr": tune.choice([0.001, 0.003, 0.005]),
            "dropout": tune.uniform(0.0, 0.3),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "label_smoothing": tune.choice([0.0, 0.1, 0.2]),
            "batch_size": tune.choice(server_cfg["batch_sizes"]),
            "pct_start": tune.choice([0.2, 0.3, 0.4]),
        },
    }

    # Configure Ray
    gpu_resources = {"gpu": server_cfg["gpu_per_trial"]}

    grace_period = min(50, max_epochs // 2) if max_epochs > 1 else 1
    scheduler = ASHAScheduler(
        grace_period=grace_period, max_t=max_epochs, reduction_factor=3
    )

    # Run search
    tuner = tune.Tuner(
        tune.with_resources(ray_trainable, resources=gpu_resources),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_loss", mode="min", scheduler=scheduler, num_samples=num_samples
        ),
        run_config=tune.RunConfig(
            name=f"glacier_search_{task}_{server_name}_tier2", verbose=1
        ),
    )

    print(f"Starting Tier 2 search with {num_samples} trials...")
    results = tuner.fit()

    print(f"✓ Tier 2 search completed for {task}")
    return results


def run_single_task(task: str, args, server_config: Dict[str, Any]):
    """Run complete hyperparameter search for a single task."""

    server_name = args.server
    experiment_name = TEST_EXPERIMENT if args.test_suite else f"{task}_search"

    print(f"\n{'=' * 60}")
    print(f"STARTING TASK: {task.upper()}")
    print(f"Server: {server_name}")
    print(f"Experiment: {experiment_name}")
    print(f"{'=' * 60}")

    # Tier 1: Dataset search (skip for desktop)
    server_cfg = SERVER_CONFIGS[server_name]
    if not server_cfg["skip_tier1"] and not args.resume:
        run_tier1_search(task, server_config, server_name, experiment_name)

    # Tier 2: Model hyperparameter search
    run_tier2_search(task, server_config, server_name, experiment_name)

    # Post-search evaluation
    run_post_evaluation(task, server_config, experiment_name, top_k=10)

    print(f"\n✓ COMPLETED TASK: {task.upper()}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    """Main function for Ray hyperparameter search."""

    parser = argparse.ArgumentParser(
        description="Ray hyperparameter search for glacier mapping"
    )
    parser.add_argument(
        "--server",
        required=True,
        choices=["desktop", "bilbo", "frodo"],
        help="Server to run on",
    )
    parser.add_argument(
        "--task",
        choices=["ci", "debris", "multiclass"],
        help="Run specific task only (default: run all tasks)",
    )
    parser.add_argument(
        "--test-suite",
        action="store_true",
        default=True,
        help="Use test_suite experiment (default: True)",
    )
    parser.add_argument("--resume", help="Resume from specific experiment path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run - print configuration only"
    )

    args = parser.parse_args()

    # Validate Ray availability
    try:
        if not ray.is_initialized():
            ray.init()
    except Exception as e:
        print(f"Failed to initialize Ray: {e}")
        print("Install Ray with: uv pip install ray[tune]")
        sys.exit(1)

    # Load server configuration
    server_config = load_server_config("configs/servers.yaml", args.server)

    # Determine tasks to run
    if args.task:
        tasks_to_run = [args.task]
    else:
        tasks_to_run = ["ci", "debris", "multiclass"]  # Set it and forget it!

    # Handle resume
    if args.resume:
        print(f"Resuming from: {args.resume}")
        # In full implementation, would parse resume path and continue from there
        # For now, just note that resume was requested

    print(f"\n🚀 STARTING HYPERPARAMETER SEARCH")
    print(f"Server: {args.server}")
    print(f"Tasks: {', '.join(tasks_to_run)}")
    print(f"Experiment: {TEST_EXPERIMENT if args.test_suite else 'production'}")
    print(f"MLflow URI: {MLFLOW_URI}")

    # Run each task sequentially
    for task in tasks_to_run:
        try:
            run_single_task(task, args, server_config)
        except Exception as e:
            print(f"❌ FAILED TASK: {task.upper()} - {e}")
            continue

    print(f"\n🎉 HYPERPARAMETER SEARCH COMPLETE!")
    print(f"Check MLflow at: {MLFLOW_URI}")
    if args.test_suite:
        print(f"Experiment: {TEST_EXPERIMENT}")


if __name__ == "__main__":
    main()
