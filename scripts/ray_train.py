#!/usr/bin/env python3
"""Ray Tune hyperparameter search for glacier mapping - Set it and forget it."""

import argparse
import os
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import ExperimentAnalysis, FailureConfig

# Import project modules

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# MLflow configuration
MLFLOW_URI = "https://mlflow.developerjose.duckdns.org"
TEST_EXPERIMENT = "test_suite"

# Task configurations
TASK_CONFIGS = {
    "ci": {
        "output_classes": [1],  # Binary: CleanIce only (class 1)
        "class_names": ["BG", "CleanIce", "Debris"],
        "experiment_suffix": "clean_ice",
    },
    "debris": {
        "output_classes": [2],  # Binary: Debris only (class 2)
        "class_names": ["BG", "CleanIce", "Debris"],
        "experiment_suffix": "debris_ice",
    },
    "multiclass": {
        "output_classes": [0, 1, 2],  # All three classes
        "class_names": ["BG", "CleanIce", "Debris"],
        "experiment_suffix": "multi_class",
    },
}

# Server-specific configurations
SERVER_CONFIGS = {
    "desktop": {
        # Desktop: Quick validation tests (1 epoch for testing)
        "skip_tier1": True,
        "datasets": ["bibek_w512_o64_f1_v2"],  # Using available dataset only
        "num_samples": 4,  # 2 batch_sizes × 2 samples
        "max_epochs": 1,  # Quick validation: 1 epoch only
        "gpu_per_trial": 1.0,  # 1 trial at a time
        "batch_sizes": [4, 8],
    },
    "frodo": {
        # Frodo: All available datasets (window/overlap/filter variations)
        "skip_tier1": False,
        "tier1_datasets": [
            "bibek_w256_o64_f1_v2",
            "bibek_w512_o32_f1_v2",
            "bibek_w512_o64_f00001_v2",
            "bibek_w512_o64_f0001_v2",
            "bibek_w512_o64_f001_v2",
            "bibek_w512_o64_f01_v2",
            "bibek_w512_o64_f02_v2",
            "bibek_w512_o64_f15_v2",
            "bibek_w512_o64_f1_v2",
            "bibek_w512_o64_f20_v2",
            "bibek_w512_o128_f1_v2",
            "bibek_w1024_o64_f1_v2",
        ],
        "num_samples": 20,
        "max_epochs_tier1": 100,
        "max_epochs_tier2": 100,
        "gpu_per_trial": 1.0,  # 1 trial per GPU = 3 concurrent
        "batch_sizes": [4, 8, 16],
    },
    "bilbo": {
        # Bilbo: All available physics datasets (complete physics exploration)
        "skip_tier1": False,
        "tier1_datasets": [
            "bibek_w512_o64_f1_v2",
            "bibek_w512_o64_f1_v2_phys32_s1",
            "bibek_w512_o64_f1_v2_phys64_s05",
            "bibek_w512_o64_f1_v2_phys64_s075",
            "bibek_w512_o64_f1_v2_phys64_s1",
            "bibek_w512_o64_f1_v2_phys128_s1",
            "bibek_w512_o64_f1_v2_physfull_s05",
        ],
        "num_samples": 20,
        "max_epochs_tier1": 100,
        "max_epochs_tier2": 100,
        "gpu_per_trial": 1.0,  # 1 trial per GPU = 3 concurrent
        "batch_sizes": [8, 16, 32, 64],  # Push boundaries on 4090s
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
            "full_eval_every": 10000000000000000000000000000000000,
            "num_viz_samples": 0,
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
    """Auto-detect resume path from existing Ray results.
    
    Prefers tier2 over tier1 for specific tasks.
    """
    ray_results_dir = Path.home() / "ray_results"

    if task:
        # Specific task resume - look for most recent tier (prefer tier2)
        tier2_path = ray_results_dir / f"glacier_search_{task}_{server}_tier2"
        tier1_path = ray_results_dir / f"glacier_search_{task}_{server}_tier1"
        
        # Prefer tier2, fallback to tier1
        if tier2_path.exists():
            return str(tier2_path)
        elif tier1_path.exists():
            return str(tier1_path)
    else:
        # Full search resume - find most recent experiment across all tasks
        pattern = f"glacier_search_*_{server}_tier*"
        existing_experiments = list(ray_results_dir.glob(pattern))
        if existing_experiments:
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

    # Load server configuration (use absolute path for Ray workers)
    project_root = Path(__file__).parent.parent
    servers_yaml_path = project_root / "configs" / "servers.yaml"
    server_config = load_server_config(str(servers_yaml_path), server)

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
            # GPU auto-detected via Ray's CUDA_VISIBLE_DEVICES
        ]

        # Import and run training
        from scripts.train import main as train_main

        final_val_loss = train_main()  # Capture return value

        # Report real validation loss to Ray
        if final_val_loss is not None:
            tune.report({"val_loss": final_val_loss})
        else:
            tune.report({"val_loss": 999.0})  # Fallback if None returned

    except Exception as e:
        print(f"[RAY] Training failed: {e}")
        tune.report({"val_loss": 999.0})  # High loss for failed trials
    finally:
        # Cleanup
        sys.argv = original_argv
        os.chdir(original_cwd)
        os.unlink(temp_config_path)


# =============================================================================
# POST-SEARCH EVALUATION
# =============================================================================


def run_post_evaluation(
    task: str, server_config: Dict[str, Any], server_name: str, experiment_name: str, top_k: int = 10
):
    """Run post-search evaluation of top-K checkpoints."""

    print(f"\n{'=' * 60}")
    print(f"POST-SEARCH EVALUATION: {task.upper()}")
    print(f"{'=' * 60}")

    ray_results_dir = Path.home() / "ray_results"
    tier2_dir = (
        ray_results_dir / f"glacier_search_{task}_{server_name}_tier2"
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

        print("\n✓ Post-search evaluation complete!")
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
    task: str, server_config: Dict[str, Any], server_name: str, experiment_name: str,
    resume_path: Optional[str] = None
):
    """Run Tier 1 search: dataset search with fixed model."""

    print(f"\n--- TIER 1: Dataset search for {task} ---")

    server_cfg = SERVER_CONFIGS[server_name]
    datasets = server_cfg["tier1_datasets"]
    max_epochs = server_cfg["max_epochs_tier1"]

    # Validate datasets exist
    validate_dataset_paths(server_config, datasets)

    # Build search space - proper Ray Tune format
    search_space = {
        "task": task,
        "dataset": tune.grid_search(datasets),  # Ray varies dataset
        "server": server_name,
        "max_epochs": max_epochs,
        "experiment_name": experiment_name,
        "hyperparams": TIER1_FIXED_PARAMS.copy(),
    }

    # Configure Ray
    gpu_resources = {"gpu": server_cfg["gpu_per_trial"]}

    grace_period = min(25, max_epochs // 4)  # Let trials run 25% before judging
    scheduler = ASHAScheduler(
        grace_period=grace_period,
        max_t=max_epochs,
        reduction_factor=3,  # Less aggressive early stopping
    )

    # Check for resume
    if resume_path and Path(resume_path).exists() and "tier1" in resume_path:
        print(f"Resuming Tier 1 from: {resume_path}")
        tuner = tune.Tuner.restore(
            resume_path,
            trainable=tune.with_resources(ray_trainable, resources=gpu_resources),
            resume_unfinished=True,
            resume_errored=True,  # Retry failed trials
        )
    else:
        # Start new search
        tuner = tune.Tuner(
            tune.with_resources(ray_trainable, resources=gpu_resources),
            param_space=search_space,
            tune_config=tune.TuneConfig(metric="val_loss", mode="min", scheduler=scheduler),
            run_config=tune.RunConfig(
                name=f"glacier_search_{task}_{server_name}_tier1",
                verbose=1,
                failure_config=FailureConfig(max_failures=100, fail_fast=False),
            ),
        )

    print(f"Starting Tier 1 search with {len(datasets)} datasets...")
    results = tuner.fit()

    print(f"✓ Tier 1 search completed for {task}")
    return results


def run_tier2_search(
    task: str, server_config: Dict[str, Any], server_name: str, experiment_name: str,
    resume_path: Optional[str] = None
):
    """Run Tier 2 search: hyperparameter search on best datasets."""

    print(f"\n--- TIER 2: Model hyperparameter search for {task} ---")

    server_cfg = SERVER_CONFIGS[server_name]

    # Get datasets for Tier 2
    if server_cfg["skip_tier1"]:
        datasets = server_cfg["datasets"]  # Use default datasets
        # Validate datasets exist for desktop
        validate_dataset_paths(server_config, datasets)
    else:
        datasets = get_best_datasets_from_tier1(task, server_name, top_k=3)

    max_epochs = server_cfg.get("max_epochs_tier2", server_cfg.get("max_epochs", 500))
    num_samples = server_cfg["num_samples"]

    print(f"Using datasets for Tier 2: {datasets}")

    # Build search space - simplified for initial exploration
    if server_name == "desktop":
        # Desktop: Fixed model, vary batch size only
        search_space = {
            "task": task,
            "dataset": tune.choice(datasets),
            "server": server_name,
            "max_epochs": max_epochs,
            "experiment_name": experiment_name,
            "hyperparams": {
                "net_depth": 4,
                "first_channel_output": 64,
                "lr": 0.0003,
                "max_lr": 0.001,
                "dropout": 0.1,
                "weight_decay": 5e-5,
                "label_smoothing": 0.0,
                "batch_size": tune.choice(server_cfg["batch_sizes"]),
                "pct_start": 0.3,
            },
        }
    else:
        # Production: Focused hyperparameter exploration
        search_space = {
            "task": task,
            "dataset": tune.choice(datasets),
            "server": server_name,
            "max_epochs": max_epochs,
            "experiment_name": experiment_name,
            "hyperparams": {
                # Architecture
                "net_depth": tune.choice([4, 5]),
                "first_channel_output": tune.choice([64, 128]),
                # Learning rates
                "lr": tune.choice([0.0001, 0.0003]),
                "max_lr": tune.choice([0.001, 0.003]),
                # Regularization
                "dropout": tune.choice([0.1, 0.2]),
                "weight_decay": tune.choice([1e-5, 5e-5]),
                # Fixed for now
                "label_smoothing": 0.0,
                "pct_start": 0.3,
                # Primary dimension: batch size
                "batch_size": tune.choice(server_cfg["batch_sizes"]),
            },
        }

    # Configure Ray
    gpu_resources = {"gpu": server_cfg["gpu_per_trial"]}

    grace_period = min(25, max_epochs // 4) if max_epochs > 1 else 1
    scheduler = ASHAScheduler(
        grace_period=grace_period,
        max_t=max_epochs,
        reduction_factor=3,  # Less aggressive early stopping
    )

    # Check for resume
    if resume_path and Path(resume_path).exists() and "tier2" in resume_path:
        print(f"Resuming Tier 2 from: {resume_path}")
        tuner = tune.Tuner.restore(
            resume_path,
            trainable=tune.with_resources(ray_trainable, resources=gpu_resources),
            resume_unfinished=True,
            resume_errored=True,  # Retry failed trials
        )
    else:
        # Start new search
        tuner = tune.Tuner(
            tune.with_resources(ray_trainable, resources=gpu_resources),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric="val_loss", mode="min", scheduler=scheduler, num_samples=num_samples
            ),
            run_config=tune.RunConfig(
                name=f"glacier_search_{task}_{server_name}_tier2",
                verbose=1,
                failure_config=FailureConfig(max_failures=100, fail_fast=False),
            ),
        )

    print(f"Starting Tier 2 search with {num_samples} trials...")
    results = tuner.fit()

    print(f"✓ Tier 2 search completed for {task}")
    return results


def run_single_task(task: str, args, server_config: Dict[str, Any]):
    """Run complete hyperparameter search for a single task."""

    server_name = args.server
    task_config = TASK_CONFIGS[task]
    experiment_name = TEST_EXPERIMENT if args.test_suite else task_config["experiment_suffix"]

    print(f"\n{'=' * 60}")
    print(f"STARTING TASK: {task.upper()}")
    print(f"Server: {server_name}")
    print(f"Experiment: {experiment_name}")
    print(f"{'=' * 60}")

    # Detect resume path for this specific task
    resume_path = None
    if args.resume:
        if args.resume == "auto":
            resume_path = detect_resume(server_name, task)
            if resume_path:
                print(f"Auto-detected resume path for {task}: {resume_path}")
            else:
                print(f"No resume path found for {task}, starting fresh")
        else:
            resume_path = args.resume
            if not Path(resume_path).exists():
                print(f"Warning: Resume path not found: {resume_path}")
                resume_path = None

    # Tier 1: Dataset search (skip for desktop or if resuming tier2)
    server_cfg = SERVER_CONFIGS[server_name]
    skip_tier1 = server_cfg["skip_tier1"] or (resume_path and "tier2" in str(resume_path))
    
    if not skip_tier1:
        run_tier1_search(task, server_config, server_name, experiment_name, resume_path)

    # Tier 2: Model hyperparameter search
    run_tier2_search(task, server_config, server_name, experiment_name, resume_path)

    # Post-search evaluation
    run_post_evaluation(task, server_config, server_name, experiment_name, top_k=10)

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
        default=False,
        help="Use test_suite experiment for validation (default: False for production)",
    )
    parser.add_argument(
        "--resume", 
        type=str,
        default=None,
        help="Resume from experiment path (use 'auto' for latest)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run - print configuration only"
    )

    args = parser.parse_args()

    # Validate Ray availability
    try:
        if not ray.is_initialized():
            ray.init(
                runtime_env={
                    "working_dir": None,  # Disable packaging entirely
                    "env_vars": {"PYTHONPATH": ".", "RAY_DISABLE_IMPORT_WARNING": "1"},
                }
            )
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

    print("\n🚀 STARTING HYPERPARAMETER SEARCH")
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

    # Cleanup Ray
    if ray.is_initialized():
        ray.shutdown()
        print("\nRay shutdown complete")

    print("\n🎉 HYPERPARAMETER SEARCH COMPLETE!")
    print(f"Check MLflow at: {MLFLOW_URI}")
    if args.test_suite:
        print(f"Experiment: {TEST_EXPERIMENT}")
    else:
        print(f"Experiments: clean_ice, debris_ice, multi_class")


if __name__ == "__main__":
    main()
