#!/usr/bin/env python3
"""Ray Tune hyperparameter search for glacier mapping - Set it and forget it."""

import argparse
import json
import os
import sys
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import ray
from ray import tune
from ray.tune import ExperimentAnalysis, FailureConfig
from ray.tune.schedulers import ASHAScheduler

# Import project modules
from glacier_mapping.utils import cleanup_gpu_memory

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
        # Desktop: Quick validation tests
        "skip_tier1": True,
        "datasets": ["bibek_w512_o64_f1_v2"],  # Using available dataset only
        "num_samples": 4,  # 2 batch_sizes × 2 samples
        "max_epochs": 500,  # Maximum epochs for desktop validation runs
        "gpu_per_trial": 1.0,  # 1 GPU = 1 trial at a time
        "batch_sizes": [4, 8],
        "num_cpus": 2,  # Limit Ray CPU usage
        "disable_mlflow": False,  # Desktop can use MLflow
    },
    "frodo": {
        # Frodo: All available datasets (window/overlap/filter variations)
        # 4x 2080 Ti GPUs (11GB each)
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
        "gpu_per_trial": 0.25,  # 4 GPUs = 4 concurrent trials max
        "batch_sizes": [4, 8, 16],
        "num_cpus": 8,  # Limit Ray CPU usage (4 trials + overhead)
        "disable_mlflow": True,  # University server: no external connections
    },
    "bilbo": {
        # Bilbo: All available physics datasets (complete physics exploration)
        # 2x 3090 GPUs (24GB each)
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
        "gpu_per_trial": 0.5,  # 2 GPUs = 2 concurrent trials max
        "batch_sizes": [8, 16, 32, 64],  # Push boundaries on 3090s
        "num_cpus": 6,  # Limit Ray CPU usage (2 trials + overhead)
        "disable_mlflow": True,  # University server: no external connections
    },
}

# Fixed Tier 1 hyperparameters (same model, different datasets)
TIER1_FIXED_PARAMS = {
    "net_depth": 4,
    "first_channel_output": 32,  # Match unet_train.yaml baseline (64 is 4x larger model)
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


def get_ray_results_dir(server_config: Dict[str, Any]) -> Path:
    """Get Ray results directory from server config (alongside other outputs)."""
    return Path(server_config["output_path"]) / "ray_results"


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
    enable_viz: bool = False,
    test_suite_mode: bool = False,
) -> Dict[str, Any]:
    """Build complete training configuration.

    Args:
        task: Task type (ci, debris, multiclass)
        dataset: Dataset name
        hyperparams: Hyperparameter dict
        server_config: Server configuration
        max_epochs: Maximum epochs to train
        enable_viz: Enable visualizations (True for test-suite/post-eval, False for HP search)
        test_suite_mode: If True, use slice_viz_every_n_epochs=1 for frequent visualization during test
    """

    task_config = TASK_CONFIGS[task]
    use_channels = get_channels_from_dataset(dataset)

    config = {
        "training_opts": {
            "dataset_name": dataset,
            "run_name": f"{task}_{dataset}_search",
            "epochs": max_epochs,
            "fine_tune": False,
            "find_lr": False,
            "early_stopping": 200
            if enable_viz
            else 100,  # Match post-eval when viz enabled
            # Visualization settings - match post-eval exactly when enable_viz=True
            "num_slice_viz": 12 if enable_viz else 0,
            "slice_viz_every_n_epochs": 1
            if test_suite_mode
            else (10 if enable_viz else 10),
            "run_full_eval": enable_viz,
            "num_full_viz": 12 if test_suite_mode else (12 if enable_viz else 0),
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


def detect_resume(
    server_config: Dict[str, Any], server: str, task: Optional[str] = None
) -> Optional[str]:
    """Auto-detect resume path from existing Ray results.

    Prefers tier2 over tier1 for specific tasks.
    """
    ray_results_dir = get_ray_results_dir(server_config)

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


def get_best_datasets_from_tier1(
    server_config: Dict[str, Any], task: str, server: str, top_k: int = 3
) -> List[str]:
    """Extract best datasets from Tier 1 results."""
    ray_results_dir = get_ray_results_dir(server_config)
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
    enable_viz = config.get("enable_viz", False)
    test_suite_mode = config.get("test_suite_mode", False)

    # Load server configuration (use absolute path for Ray workers)
    project_root = Path(__file__).parent.parent
    servers_yaml_path = project_root / "configs" / "servers.yaml"
    server_config = load_server_config(str(servers_yaml_path), server)
    server_cfg = SERVER_CONFIGS.get(server, {})

    # Build complete training configuration
    full_config = build_base_config(
        task,
        dataset,
        hyperparams,
        server_config,
        max_epochs,
        enable_viz=enable_viz,
        test_suite_mode=test_suite_mode,
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
            "--output-dir",
            server_config["output_path"],
            # GPU auto-detected via Ray's CUDA_VISIBLE_DEVICES
        ]

        # Conditional flags based on mode and server config
        if enable_viz:
            # Post-eval or test-suite mode: enable outputs
            if server_cfg.get("disable_mlflow", False):
                # University server: no MLflow, but keep local outputs
                sys.argv.extend(["--mlflow-enabled", "false"])
                print(f"[RAY] MLflow disabled for {server} (local logging only)")
            else:
                # Desktop: enable MLflow
                sys.argv.extend(
                    ["--mlflow-enabled", "true", "--tracking-uri", MLFLOW_URI]
                )
        else:
            # Hyperparameter search mode: disable MLflow, skip disk writes
            sys.argv.extend(["--mlflow-enabled", "false", "--no-output"])

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
        # GPU cleanup - critical for preventing OOM in subsequent trials
        cleanup_gpu_memory()

        # Cleanup
        sys.argv = original_argv
        os.chdir(original_cwd)
        if Path(temp_config_path).exists():
            os.unlink(temp_config_path)


# =============================================================================
# POST-SEARCH EVALUATION
# =============================================================================


def run_post_evaluation(
    task: str,
    server_config: Dict[str, Any],
    server_name: str,
    experiment_name: str,
    top_k: int = 3,
    search_space_info: Optional[Dict[str, Any]] = None,
):
    """Run full training of top-K best configurations from hyperparameter search.

    Trains the top-K hyperparameter configurations discovered during search with:
    - Full max_epochs training (from server config)
    - Early stopping (patience=200 epochs)
    - Enhanced visualizations (12 tiles: top/middle/bottom IoU)
    - Complete MLflow logging and checkpointing

    Args:
        task: Task type (ci, debris, multiclass)
        server_config: Server configuration dict
        server_name: Server name string
        experiment_name: MLflow experiment name
        top_k: Number of top trials to fully train (default: 3)
        search_space_info: Search space configuration for metadata
    """

    # Aggressive GPU cleanup before starting post-evaluation
    print("\n🧹 Aggressive GPU cleanup before post-evaluation...")
    cleanup_gpu_memory()

    print(f"\n{'=' * 60}")
    print(f"POST-EVALUATION: {task.upper()}")
    print(f"Training top-{top_k} configurations with full epochs")
    print(f"{'=' * 60}")

    ray_results_dir = get_ray_results_dir(server_config)

    # Find results directory (prefer tier2, fallback to test_suite)
    tier2_dir = ray_results_dir / f"glacier_search_{task}_{server_name}_tier2"
    test_suite_dir = ray_results_dir / f"glacier_test_suite_{task}_{server_name}"

    if tier2_dir.exists():
        results_dir = tier2_dir
        print(f"Using tier2 results: {tier2_dir.name}")
    elif test_suite_dir.exists():
        results_dir = test_suite_dir
        print(f"Using test_suite results: {test_suite_dir.name}")
    else:
        print(f"❌ No results found for {task}, skipping post-evaluation")
        return

    try:
        # Load Ray analysis and get results dataframe
        analysis = ExperimentAnalysis(str(results_dir))

        # Get all results as dataframe
        try:
            results_df = analysis.dataframe()
        except Exception as e:
            print(f"Failed to load results dataframe: {e}")
            # Fallback: use get_best_trial
            best_trial = analysis.get_best_trial(metric="val_loss", mode="min")
            if not best_trial:
                print("No trials found in results")
                return
            # Create minimal dataframe from best trial
            results_df = pd.DataFrame(
                [
                    {
                        "trial_id": best_trial.trial_id,
                        "val_loss": best_trial.last_result.get("val_loss", 999.0),
                        **{f"config/{k}": v for k, v in best_trial.config.items()},
                    }
                ]
            )

        if results_df.empty:
            print(f"No trials found in {results_dir}")
            return

        # Sort by val_loss and get top-K
        results_df = results_df.sort_values("val_loss", ascending=True)
        top_trials = results_df.head(top_k)

        print(f"\n✓ Found {len(results_df)} total trials")
        print(f"✓ Will train top-{len(top_trials)} configurations\n")

        # Get max_epochs from server config
        server_cfg = SERVER_CONFIGS[server_name]
        max_epochs_final = server_cfg.get(
            "max_epochs_tier2", server_cfg.get("max_epochs", 500)
        )

        print("Post-evaluation settings:")
        print(f"  Max epochs: {max_epochs_final}")
        print("  Early stopping patience: 200 epochs")
        print("  Slice visualizations: every 10 epochs")
        print("  Full evaluations: on model improvement (12 tiles)")

        trained_results = []

        # Train each top configuration sequentially
        for rank, (idx, trial_row) in enumerate(top_trials.iterrows(), start=1):
            trial_id = trial_row.get("trial_id", f"trial_{idx}")
            search_val_loss = trial_row.get("val_loss", 999.0)

            print(f"\n{'─' * 60}")
            print(f"RANK {rank}/{top_k}: {trial_id}")
            print(f"Search validation loss: {search_val_loss:.4f}")
            print(f"{'─' * 60}")

            # Extract hyperparameters from trial config columns
            hyperparams = {}
            dataset = None

            for col in trial_row.index:
                col_str = str(col)  # Ensure column name is string
                if col_str.startswith("config/hyperparams/"):
                    param_name = col_str.replace("config/hyperparams/", "")
                    hyperparams[param_name] = trial_row[col]
                elif col_str == "config/dataset":
                    dataset = str(trial_row[col])

            if not dataset:
                print("⚠️  Could not extract dataset from trial config, skipping")
                continue

            print(f"Dataset: {dataset}")
            print(f"Hyperparameters: {hyperparams}")

            # Build full training configuration
            # enable_viz=True, test_suite_mode=False gives us the production post-eval settings:
            # - num_full_viz=12 (top/middle/bottom IoU selection)
            # - slice_viz_every_n_epochs=10
            # - early_stopping=200
            # - run_full_eval=True
            full_config = build_base_config(
                task=task,
                dataset=dataset,
                hyperparams=hyperparams,
                server_config=server_config,
                max_epochs=max_epochs_final,
                enable_viz=True,  # Enable all visualizations
                test_suite_mode=False,  # Use production settings (slice viz every 10 epochs)
            )

            # Modify run_name to indicate post-evaluation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_config["training_opts"]["run_name"] = (
                f"posteval_rank{rank}_{task}_{dataset}_{timestamp}"
            )

            # Create temporary config file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(full_config, f)
                temp_config_path = f.name

            # Run full training
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
                    server_name,
                    "--max-epochs",
                    str(max_epochs_final),
                    "--experiment-name",
                    experiment_name,
                    "--output-dir",
                    server_config["output_path"],
                ]

                # Add MLflow args only if not disabled
                server_cfg = SERVER_CONFIGS.get(server_name, {})
                if not server_cfg.get("disable_mlflow", False):
                    sys.argv.extend(
                        [
                            "--tracking-uri",
                            MLFLOW_URI,
                            "--mlflow-enabled",
                            "true",
                        ]
                    )
                    print(f"  MLflow enabled: {MLFLOW_URI}")
                else:
                    sys.argv.extend(["--mlflow-enabled", "false"])
                    print(f"  MLflow disabled for {server_name} (local logging only)")

                # Import and run training
                from scripts.train import main as train_main

                print(
                    f"\n🚀 Starting full training (max {max_epochs_final} epochs, early stop @ 200)..."
                )
                final_val_loss = train_main()

                # Collect results
                result = {
                    "rank": rank,
                    "trial_id": trial_id,
                    "dataset": dataset,
                    "search_val_loss": search_val_loss,
                    "final_val_loss": final_val_loss if final_val_loss else 999.0,
                    "hyperparameters": hyperparams,
                    "run_name": full_config["training_opts"]["run_name"],
                }
                trained_results.append(result)

                print(f"\n✓ Rank {rank} training complete!")
                print(f"  Final validation loss: {result['final_val_loss']:.4f}")
                print(f"  Run name: {result['run_name']}")

            except Exception as e:
                print(f"\n❌ Training failed for rank {rank}: {e}")
                import traceback

                traceback.print_exc()

                result = {
                    "rank": rank,
                    "trial_id": trial_id,
                    "dataset": dataset,
                    "search_val_loss": search_val_loss,
                    "final_val_loss": 999.0,
                    "hyperparameters": hyperparams,
                    "run_name": full_config["training_opts"]["run_name"],
                    "error": str(e),
                }
                trained_results.append(result)

            finally:
                # Cleanup
                sys.argv = original_argv
                os.chdir(original_cwd)
                if Path(temp_config_path).exists():
                    os.unlink(temp_config_path)

                # GPU cleanup between trainings
                print("\n🧹 Cleaning GPU memory before next training...")
                cleanup_gpu_memory()

        # Save comparison results
        if not trained_results:
            print("\n⚠️  No models were successfully trained")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path(server_config["output_path"])
            / "hyperparameter_search_results"
            / experiment_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_filename = f"{task}_{server_name}_{timestamp}_top{top_k}_trained.csv"
        comparison_path = output_dir / csv_filename

        # Create comprehensive CSV
        csv_data = []
        for result in trained_results:
            row = {
                "rank": result["rank"],
                "trial_id": result["trial_id"],
                "dataset": result["dataset"],
                "search_val_loss": result["search_val_loss"],
                "final_val_loss": result["final_val_loss"],
                "run_name": result.get("run_name", ""),
            }
            row.update(result["hyperparameters"])
            if "error" in result:
                row["error"] = result["error"]
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        df.to_csv(comparison_path, index=False)

        print(f"\n{'=' * 60}")
        print("POST-EVALUATION COMPLETE!")
        print(f"{'=' * 60}")
        print(f"✓ Results saved to: {comparison_path}")

        # Save metadata
        best_result = trained_results[0] if trained_results else None
        metadata = {
            "task": task,
            "server": server_name,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "top_k": top_k,
            "max_epochs_final": max_epochs_final,
            "early_stopping_patience": 200,
            "results_dir": str(results_dir),
            "search_space": search_space_info or {},
            "trained_models": len(trained_results),
            "successful_trainings": len(
                [r for r in trained_results if "error" not in r]
            ),
            "failed_trainings": len([r for r in trained_results if "error" in r]),
            "best_model": {
                "rank": 1,
                "trial_id": best_result["trial_id"],
                "dataset": best_result["dataset"],
                "search_val_loss": best_result["search_val_loss"],
                "final_val_loss": best_result["final_val_loss"],
                "hyperparameters": best_result["hyperparameters"],
                "run_name": best_result.get("run_name", ""),
            }
            if best_result and "error" not in best_result
            else None,
        }

        metadata_path = output_dir / f"{task}_{server_name}_{timestamp}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to: {metadata_path}")

        # Print summary
        print("\nTraining Summary:")
        print(f"  Total models: {len(trained_results)}")
        print(f"  Successful: {metadata['successful_trainings']}")
        print(f"  Failed: {metadata['failed_trainings']}")

        # Print best model recommendation
        if best_result and "error" not in best_result:
            print(f"\n🏆 BEST TRAINED MODEL ({task.upper()}):")
            print(f"  Run: {best_result.get('run_name')}")
            print(f"  Dataset: {best_result['dataset']}")
            print(f"  Search val_loss: {best_result['search_val_loss']:.4f}")
            print(f"  Final val_loss: {best_result['final_val_loss']:.4f}")
            print("  Hyperparameters:")
            for k, v in best_result["hyperparameters"].items():
                print(f"    {k}: {v}")

    except Exception as e:
        print(f"\n❌ Post-evaluation failed: {e}")
        import traceback

        traceback.print_exc()


# =============================================================================
# SEARCH EXECUTION FUNCTIONS
# =============================================================================


def run_tier1_search(
    task: str,
    server_config: Dict[str, Any],
    server_name: str,
    experiment_name: str,
    resume_path: Optional[str] = None,
):
    """Run Tier 1 search: dataset search with fixed model."""

    print(f"\n--- TIER 1: Dataset search for {task} ---")

    server_cfg = SERVER_CONFIGS[server_name]
    datasets = server_cfg["tier1_datasets"]
    max_epochs = server_cfg["max_epochs_tier1"]
    ray_results_dir = get_ray_results_dir(server_config)

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
            tune_config=tune.TuneConfig(
                metric="val_loss", mode="min", scheduler=scheduler
            ),
            run_config=tune.RunConfig(
                name=f"glacier_search_{task}_{server_name}_tier1",
                storage_path=str(ray_results_dir),
                verbose=1,
                failure_config=FailureConfig(max_failures=100, fail_fast=False),
            ),
        )

    print(f"Ray results will be saved to: {ray_results_dir}")
    print(f"Starting Tier 1 search with {len(datasets)} datasets...")
    results = tuner.fit()

    print(f"✓ Tier 1 search completed for {task}")
    return results


def run_tier2_search(
    task: str,
    server_config: Dict[str, Any],
    server_name: str,
    experiment_name: str,
    resume_path: Optional[str] = None,
):
    """Run Tier 2 search: hyperparameter search on best datasets."""

    print(f"\n--- TIER 2: Model hyperparameter search for {task} ---")

    server_cfg = SERVER_CONFIGS[server_name]
    ray_results_dir = get_ray_results_dir(server_config)

    # Get datasets for Tier 2
    if server_cfg["skip_tier1"]:
        datasets = server_cfg.get("datasets", server_cfg.get("tier1_datasets", []))
        # Validate datasets exist for desktop
        validate_dataset_paths(server_config, datasets)
    else:
        datasets = get_best_datasets_from_tier1(
            server_config, task, server_name, top_k=3
        )

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
        # Capture search space info for metadata
        search_space_info = {
            "datasets": datasets,
            "batch_sizes": server_cfg["batch_sizes"],
            "hyperparams": "fixed (desktop mode)",
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
        # Capture search space info for metadata
        search_space_info = {
            "datasets": datasets,
            "batch_sizes": server_cfg["batch_sizes"],
            "net_depth": [4, 5],
            "first_channel_output": [64, 128],
            "lr": [0.0001, 0.0003],
            "max_lr": [0.001, 0.003],
            "dropout": [0.1, 0.2],
            "weight_decay": [1e-5, 5e-5],
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
                metric="val_loss",
                mode="min",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            run_config=tune.RunConfig(
                name=f"glacier_search_{task}_{server_name}_tier2",
                storage_path=str(ray_results_dir),
                verbose=1,
                failure_config=FailureConfig(max_failures=100, fail_fast=False),
            ),
        )

    print(f"Ray results will be saved to: {ray_results_dir}")
    print(f"Starting Tier 2 search with {num_samples} trials...")
    results = tuner.fit()

    print(f"✓ Tier 2 search completed for {task}")
    return results, search_space_info


def run_single_task(task: str, args, server_config: Dict[str, Any]):
    """Run complete hyperparameter search for a single task."""

    server_name = args.server
    task_config = TASK_CONFIGS[task]
    experiment_name = task_config["experiment_suffix"]

    print(f"\n{'=' * 60}")
    print(f"STARTING TASK: {task.upper()}")
    print(f"Server: {server_name}")
    print(f"Experiment: {experiment_name}")
    print(f"{'=' * 60}")

    # Detect resume path for this specific task
    resume_path = None
    if args.resume:
        if args.resume == "auto":
            resume_path = detect_resume(server_config, server_name, task)
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
    skip_tier1 = server_cfg["skip_tier1"] or (
        resume_path and "tier2" in str(resume_path)
    )

    if not skip_tier1:
        run_tier1_search(task, server_config, server_name, experiment_name, resume_path)

    # Tier 2: Model hyperparameter search
    _, search_space_info = run_tier2_search(
        task, server_config, server_name, experiment_name, resume_path
    )

    # GPU cleanup after search completes, before post-evaluation
    print("\n🧹 Cleaning up GPU memory before post-evaluation...")
    cleanup_gpu_memory()

    # Post-search evaluation
    run_post_evaluation(
        task,
        server_config,
        server_name,
        experiment_name,
        top_k=3,  # Train top-3 configurations
        search_space_info=search_space_info,
    )

    print(f"\n✓ COMPLETED TASK: {task.upper()}")


def run_test_suite(task: str, server_config: Dict[str, Any], server_name: str):
    """Run test suite: quick validation followed by full training with visualizations.

    This mode runs two phases:
    1. Quick validation (2 epochs) to test the Ray -> train.py pipeline
    2. Full training with all visualizations enabled (top-1 post-evaluation)

    Ideal for:
    - Testing the full training pipeline end-to-end
    - Getting a complete trained model with comprehensive visualizations
    - CI/validation workflows

    Args:
        task: Task type (ci, debris, multiclass)
        server_config: Server configuration dict
        server_name: Server name string
    """

    server_cfg = SERVER_CONFIGS[server_name]

    print(f"\n{'=' * 60}")
    print(f"TEST SUITE: {task.upper()}")
    print(f"Server: {server_name}")
    print("\nPHASE 1: Quick validation")
    print("  Epochs: 2")
    print("  Visualizations: ENABLED (12 full-tile)")
    print("  Slice viz frequency: Every 1 epoch")
    print("\nPHASE 2: Full training (runs after validation)")
    print(
        f"  Epochs: {server_cfg.get('max_epochs_tier2', server_cfg.get('max_epochs', 500))}"
    )
    print("  Early stopping: 200 epochs")
    print("  Slice viz frequency: Every 10 epochs")
    print("  Full evaluations: On best model improvement (12 tiles)")
    print("\nMLflow: ENABLED")
    print(f"{'=' * 60}")
    ray_results_dir = get_ray_results_dir(server_config)

    # Get dataset - use first available
    datasets = server_cfg.get("datasets", server_cfg.get("tier1_datasets", []))
    if not datasets:
        raise ValueError(f"No datasets configured for server {server_name}")
    dataset = datasets[0]

    # Validate dataset exists
    validate_dataset_paths(server_config, [dataset])

    # Fixed hyperparams for test suite (no search)
    hyperparams = TIER1_FIXED_PARAMS.copy()

    # Config for single trial with viz enabled
    param_space = {
        "task": task,
        "dataset": dataset,  # Single dataset, not tune.choice
        "server": server_name,
        "max_epochs": 2,
        "experiment_name": TEST_EXPERIMENT,
        "enable_viz": True,  # Enable all visualizations (12 tiles)
        "test_suite_mode": True,  # Use slice_viz_every_n_epochs=1 for test
        "hyperparams": hyperparams,
    }

    # Configure Ray - single trial, no scheduler
    gpu_resources = {"gpu": server_cfg["gpu_per_trial"]}

    tuner = tune.Tuner(
        tune.with_resources(ray_trainable, resources=gpu_resources),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=1,  # Single trial
            # No scheduler - let it run to completion
        ),
        run_config=tune.RunConfig(
            name=f"glacier_test_suite_{task}_{server_name}",
            storage_path=str(ray_results_dir),
            verbose=1,
        ),
    )

    print(f"Ray results will be saved to: {ray_results_dir}")
    print(f"Dataset: {dataset}")
    print("Starting PHASE 1: Quick validation (2 epochs)...")
    results = tuner.fit()

    print("\n✓ Phase 1 completed: Quick validation successful")

    # Capture search space for metadata
    experiment_name = TEST_EXPERIMENT
    search_space_info = {
        "datasets": [dataset],
        "hyperparams": "fixed (test suite mode)",
        "mode": "test_suite",
    }

    # GPU cleanup before post-evaluation
    print("\n🧹 Cleaning up GPU memory before post-evaluation...")
    cleanup_gpu_memory()

    # Run post-evaluation with top-1 (only 1 trial exists)
    print("\n" + "=" * 60)
    print("STARTING PHASE 2: Full training with visualizations")
    print("=" * 60)

    run_post_evaluation(
        task=task,
        server_config=server_config,
        server_name=server_name,
        experiment_name=experiment_name,
        top_k=1,  # Test suite only has 1 trial
        search_space_info=search_space_info,
    )

    return results


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
        help="Run single 2-epoch test with visualizations (no HP search). Requires --task.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from experiment path (use 'auto' for latest)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run - print configuration only"
    )

    args = parser.parse_args()

    # Validate --test-suite requires --task
    if args.test_suite and not args.task:
        parser.error("--test-suite requires --task to be specified")

    # Load server configuration FIRST (needed for Ray init)
    server_config = load_server_config("configs/servers.yaml", args.server)
    server_cfg = SERVER_CONFIGS.get(args.server, {})

    # Initialize Ray with server-specific resource limits (quiet mode for universities)
    try:
        if not ray.is_initialized():
            ray.init(
                num_cpus=server_cfg.get("num_cpus", 4),  # Limit CPU usage
                include_dashboard=False,  # No web dashboard (saves ports)
                configure_logging=False,  # Reduce logging overhead
                runtime_env={
                    "working_dir": None,  # Disable packaging entirely
                    "env_vars": {
                        "PYTHONPATH": ".",
                        "RAY_DISABLE_IMPORT_WARNING": "1",
                        "RAY_DISABLE_DASHBOARD": "1",
                    },
                },
            )
            print(
                f"✓ Ray initialized: {server_cfg.get('num_cpus', 4)} CPUs, dashboard disabled"
            )
            if server_cfg.get("disable_mlflow", False):
                print(
                    f"✓ MLflow disabled for {args.server} (university server mode - local logging only)"
                )
    except Exception as e:
        print(f"Failed to initialize Ray: {e}")
        print("Install Ray with: uv pip install ray[tune]")
        sys.exit(1)

    # Handle test-suite mode separately
    if args.test_suite:
        print("\n🧪 STARTING TEST SUITE")
        print(f"Server: {args.server}")
        print(f"Task: {args.task}")
        print(f"Experiment: {TEST_EXPERIMENT}")
        if not server_cfg.get("disable_mlflow", False):
            print(f"MLflow URI: {MLFLOW_URI}")
        else:
            print("MLflow: DISABLED (local logging only)")

        try:
            run_test_suite(args.task, server_config, args.server)
        except Exception as e:
            print(f"❌ TEST SUITE FAILED: {e}")
            raise

        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()
            print("\nRay shutdown complete")

        print("\n🎉 TEST SUITE COMPLETE!")
        print("✓ Phase 1 - Quick validation (2 epochs): PASSED")
        print("✓ Phase 2 - Full training with visualizations: COMPLETED")
        if not server_cfg.get("disable_mlflow", False):
            print(f"\nCheck MLflow at: {MLFLOW_URI}")
            print(f"Experiment: {TEST_EXPERIMENT}")
        else:
            print(f"\nResults saved locally to: {server_config['output_path']}")
            print("Use rsync to copy to desktop, then upload with upload_to_mlflow.py")
        print(f"Look for runs starting with: posteval_rank1_{args.task}_")
        return

    # Normal hyperparameter search mode
    # Determine tasks to run
    if args.task:
        tasks_to_run = [args.task]
    else:
        tasks_to_run = ["ci", "debris", "multiclass"]  # Set it and forget it!

    print("\n🚀 STARTING HYPERPARAMETER SEARCH")
    print(f"Server: {args.server}")
    print(f"Tasks: {', '.join(tasks_to_run)}")
    print("Experiment: production")
    if not server_cfg.get("disable_mlflow", False):
        print(f"MLflow URI: {MLFLOW_URI}")
    else:
        print("MLflow: DISABLED (local logging only)")

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
    if not server_cfg.get("disable_mlflow", False):
        print(f"Check MLflow at: {MLFLOW_URI}")
        print("Experiments: clean_ice, debris_ice, multi_class")
    else:
        print(f"Results saved locally to: {server_config['output_path']}")
        print("Use rsync to copy to desktop, then upload with upload_to_mlflow.py")


if __name__ == "__main__":
    main()
