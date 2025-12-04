#!/usr/bin/env python3
"""Upload completed training runs to MLflow with checkpoint evaluation."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional


try:
    from mlflow.store.artifact.artifact_repository_registry import (
        get_artifact_repository,
    )

    ARTIFACT_REGISTRY_AVAILABLE = True
except ImportError:
    ARTIFACT_REGISTRY_AVAILABLE = False


try:
    import glacier_mapping.utils.mlflow_utils as mlflow_utils

    MLFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Error: MLflow utilities not available: {e}")
    MLFLOW_AVAILABLE = False

try:
    from tensorboard.backend.event_processing import event_file_loader

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
    from glacier_mapping.lightning.callbacks import ValidationVisualizationCallback
    from glacier_mapping.lightning.best_model_callback import TestEvaluationCallback
    from glacier_mapping.lightning.glacier_datamodule import GlacierDataModule

    LIGHTNING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Lightning modules not available: {e}")
    LIGHTNING_AVAILABLE = False


def parse_tensorboard_logs(log_dir: Path) -> Dict[str, float]:
    """Parse TensorBoard logs to extract metrics."""
    if not TENSORBOARD_AVAILABLE:
        print("Warning: TensorBoard not available, cannot parse logs")
        return {}

    metrics: Dict[str, float] = {}
    try:
        # Find all event files
        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            print(f"Warning: No TensorBoard event files found in {log_dir}")
            return metrics

        # Load events from the most recent file
        event_file = max(event_files, key=os.path.getctime)

        for event in event_file_loader.EventFileLoader(str(event_file)).Load():  # type: ignore[misc]
            if not hasattr(event, "summary") or not hasattr(event.summary, "value"):  # type: ignore[attr-defined]
                continue

            for value in event.summary.value:  # type: ignore[attr-defined]
                # Handle different TensorBoard protobuf formats
                tag = getattr(value, "tag", None)
                simple_value = getattr(value, "simple_value", None)

                if tag is None or simple_value is None:
                    continue

                # For best metrics, keep the best value; for regular metrics, keep latest
                if "best_" in tag:
                    if tag not in metrics or simple_value > metrics[tag]:
                        metrics[tag] = simple_value
                elif tag in ["epoch", "val_loss", "train_loss_epoch"]:
                    # Keep the latest for these important metrics
                    metrics[tag] = simple_value
                else:
                    # For other metrics, keep latest
                    metrics[tag] = simple_value

        return metrics

    except Exception as e:
        print(f"Warning: Failed to parse TensorBoard logs: {e}")
        return {}


def delete_mlflow_artifact_directories(run_id: str, artifact_paths: list[str]) -> None:
    """
    Delete artifact directories from MLflow run before re-upload.

    Args:
        run_id: MLflow run ID
        artifact_paths: List of artifact directory paths to delete
                       (e.g., ["val_visualizations", "test_evaluations"])
    """
    if not MLFLOW_AVAILABLE:
        print("  Warning: MLflow not available, cannot delete artifacts")
        return

    if not ARTIFACT_REGISTRY_AVAILABLE:
        print("  Warning: Artifact registry not available, cannot delete artifacts")
        return

    try:
        import mlflow

        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri

        if not artifact_uri:
            print("  Warning: No artifact URI found")
            return

        print(f"  Artifact URI: {artifact_uri}")

        # Get artifact repository
        repository = get_artifact_repository(artifact_uri)  # type: ignore[misc]

        # Delete each artifact directory
        for artifact_path in artifact_paths:
            try:
                # List artifacts to check if path exists
                artifacts = client.list_artifacts(run_id, artifact_path)
                if artifacts:
                    print(f"  Deleting old artifacts: {artifact_path}/")
                    # Delete the entire directory
                    repository.delete_artifacts(artifact_path)
                else:
                    print(f"  No existing artifacts at: {artifact_path}/ (skipping)")
            except Exception:
                # Path doesn't exist or already deleted - this is fine
                print(f"  No artifacts to delete at {artifact_path}/ (new upload)")

    except Exception as e:
        print(f"  Warning: Could not delete artifacts: {e}")
        print("  Continuing with upload (may result in duplicate artifacts)")


def load_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load configuration from saved run."""
    config_path = run_dir / "conf.json"
    if not config_path.exists():
        print(f"Warning: No config found at {config_path}")
        return None

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return None


def detect_experiment_type(config: Dict[str, Any]) -> str:
    """Detect experiment type from config using mlflow_utils."""
    if not MLFLOW_AVAILABLE:
        return "unknown"

    return mlflow_utils.categorize_experiment(config)  # type: ignore[misc]


def extract_epoch_from_checkpoint(
    checkpoint_path: Path, fallback_index: int = 0
) -> int:
    """Extract epoch number from checkpoint filename.

    Args:
        checkpoint_path: Path to checkpoint file
        fallback_index: Index to use as fallback if epoch cannot be extracted

    Returns:
        Epoch number (int), or fallback_index if extraction fails

    Examples:
        - name_epoch=123_val_loss=0.1234.ckpt -> 123
        - name_epoch123_val_loss0.1234.ckpt -> 123
        - unknown_name.ckpt -> fallback_index
    """
    import re

    match = re.search(r"epoch[=_](\d+)", checkpoint_path.name)
    if match:
        return int(match.group(1))
    else:
        # Use fallback with checkpoint name as identifier
        print(
            f"  Warning: Cannot extract epoch from {checkpoint_path.name}, using index {fallback_index}"
        )
        return fallback_index


def find_all_checkpoints(run_dir: Path) -> list[Path]:
    """Find all checkpoints in the run directory.

    Returns:
        List of checkpoint paths, sorted by epoch number (oldest to newest)
    """
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"Warning: No checkpoints directory found in {run_dir}")
        return []

    # Look for epoch-based checkpoints (exclude last.ckpt)
    checkpoint_files = [
        f for f in checkpoint_dir.glob("*.ckpt") if f.name != "last.ckpt"
    ]

    if not checkpoint_files:
        print(f"Warning: No checkpoint files found in {checkpoint_dir}")
        return []

    # Sort by epoch number extracted from filename
    checkpoint_files.sort(key=lambda p: extract_epoch_from_checkpoint(p, 0))
    return checkpoint_files


def regenerate_visualizations(
    run_dir: Path,
    config: Dict[str, Any],
    checkpoint_path: Path,
    checkpoint_epoch: int,
    server_config: Dict[str, Any],
    high_res: bool = False,
    val_viz_n: int = 4,
    test_eval_n: int = 4,
) -> tuple[bool, Dict[str, float]]:
    """Regenerate validation and test visualizations from checkpoint.

    Args:
        run_dir: Run directory
        config: Run configuration
        checkpoint_path: Path to checkpoint file
        checkpoint_epoch: Epoch number for this checkpoint (used in filenames)
        server_config: Server configuration dict (includes image_dir)
        high_res: Generate high-resolution visualizations
        val_viz_n: Number of validation visualizations per third (total = 3 * n)
        test_eval_n: Number of test visualizations per third (total = 3 * n)

    Returns:
        Tuple of (success: bool, metrics: Dict[str, float])
    """
    if not LIGHTNING_AVAILABLE:
        print("Error: Lightning modules not available for visualization generation")
        return False, {}

    try:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        if not LIGHTNING_AVAILABLE:
            print("Error: Lightning modules not available")
            return False, {}

        model = GlacierSegmentationModule.load_from_checkpoint(str(checkpoint_path))  # type: ignore[misc]
        model.eval()

        # Ensure visualization directories exist (don't clear - allow accumulation)
        val_viz_dir = run_dir / "val_visualizations"
        test_eval_dir = run_dir / "test_evaluations"

        val_viz_dir.mkdir(parents=True, exist_ok=True)
        test_eval_dir.mkdir(parents=True, exist_ok=True)

        # Initialize datamodule
        loader_opts = config.get("loader_opts", {})
        processed_dir = loader_opts.get("processed_dir", "data/processed")

        datamodule = GlacierDataModule(  # type: ignore[misc]
            processed_dir=processed_dir,
            batch_size=loader_opts.get("batch_size", 8),
            landsat_channels=loader_opts.get("landsat_channels", True),
            dem_channels=loader_opts.get("dem_channels", True),
            spectral_indices_channels=loader_opts.get(
                "spectral_indices_channels", True
            ),
            hsv_channels=loader_opts.get("hsv_channels", True),
            physics_channels=loader_opts.get("physics_channels", False),
            velocity_channels=loader_opts.get("velocity_channels", True),
            output_classes=loader_opts.get("output_classes", [0, 1, 2]),
            class_names=loader_opts.get("class_names", ["BG", "CleanIce", "Debris"]),
            normalize=loader_opts.get("normalize", "mean-std"),
        )
        datamodule.setup("fit")

        # Create a minimal trainer (no actual training, just for callback context)
        class MinimalTrainer:
            """Minimal trainer mock for callbacks."""

            def __init__(self, run_dir: Path, epoch: int):
                # Callbacks use `current_epoch + 1` for filenames, so subtract 1 here
                # to match the checkpoint epoch number
                self.current_epoch = epoch - 1
                self.sanity_checking = False
                self.callback_metrics = {}
                self.loggers = []
                self.checkpoint_callback = None
                self.default_root_dir = str(run_dir)

        trainer = MinimalTrainer(run_dir, checkpoint_epoch)

        # Generate validation visualizations
        print(f"Generating validation visualizations (n={val_viz_n})...")
        val_callback = ValidationVisualizationCallback(  # type: ignore[misc]
            viz_n=val_viz_n,
            log_every_n_epochs=1,  # Force generation
            selection="iou",
            save_dir=str(val_viz_dir),
            image_dir=server_config.get("image_dir"),
        )
        val_callback._generate_visualizations(trainer, model)  # type: ignore[arg-type]
        print(f"✅ Validation visualizations saved to {val_viz_dir}")

        # Generate test evaluations
        print(f"Generating test evaluations (n={test_eval_n})...")
        test_callback = TestEvaluationCallback(  # type: ignore[misc]
            viz_n=test_eval_n, image_dir=server_config.get("image_dir")
        )

        # Manually trigger test evaluation (bypass validation_epoch_end)
        test_callback._run_full_evaluation(trainer, model)  # type: ignore[arg-type]
        print(f"✅ Test evaluations saved to {test_eval_dir}")

        # Extract new best test metrics from the callback
        new_metrics = {}
        for class_name, metrics in test_callback.best_test_metrics.items():
            new_metrics[f"best_test_{class_name}_iou"] = metrics["iou"]
            new_metrics[f"best_test_{class_name}_precision"] = metrics["precision"]
            new_metrics[f"best_test_{class_name}_recall"] = metrics["recall"]

        return True, new_metrics

    except Exception as e:
        print(f"❌ Failed to regenerate visualizations: {e}")
        import traceback

        traceback.print_exc()
        return False, {}


def upload_single_run(
    run_dir: Path,
    tracking_uri: str,
    server_name: str,
    regenerate: bool = False,
    high_res: bool = False,
    val_viz_n: int = 4,
    test_eval_n: int = 4,
) -> bool:
    """Upload a single run to MLflow.

    Args:
        run_dir: Path to run directory
        tracking_uri: MLflow tracking URI
        server_name: Server name for loading image_dir path
        regenerate: Regenerate visualizations and update existing run
        high_res: Generate high-resolution visualizations
        val_viz_n: Number of validation visualizations per third
        test_eval_n: Number of test visualizations per third

    Returns:
        True if successful, False otherwise
    """
    print(f"Processing run: {run_dir.name}")

    # Load server configuration
    import yaml

    servers_yaml_path = Path("configs/servers.yaml")
    if not servers_yaml_path.exists():
        print(f"Error: servers.yaml not found at {servers_yaml_path}")
        return False

    with open(servers_yaml_path, "r") as f:
        servers = yaml.safe_load(f)

    if server_name not in servers:
        print(f"Error: Server '{server_name}' not found in servers.yaml")
        return False

    server_config = servers[server_name]
    print(f"Using server config: {server_name}")
    print(f"  Image directory: {server_config.get('image_dir')}")

    # Load configuration
    config = load_run_config(run_dir)
    if not config:
        print(f"Skipping {run_dir.name}: No configuration found")
        return False

    # Parse TensorBoard logs for existing metrics (skip if regenerating)
    metrics: Dict[str, float] = {}

    if not regenerate:
        log_dir = run_dir / "logs"
        if not log_dir.exists():
            print(f"Skipping {run_dir.name}: No logs directory found")
            return False

        # Find the actual log subdirectory
        log_subdirs = [d for d in log_dir.iterdir() if d.is_dir()]
        if not log_subdirs:
            print(f"Skipping {run_dir.name}: No log subdirectories found")
            return False

        # Use the most recent log directory
        log_dir = max(log_subdirs, key=os.path.getctime)

        metrics = parse_tensorboard_logs(log_dir)
        if not metrics:
            print(f"Skipping {run_dir.name}: No metrics found")
            return False
    else:
        print("Regenerate mode: will only upload new metrics and visualizations")

    # Regenerate visualizations if requested
    all_checkpoint_metrics: Dict[str, float] = {}
    if regenerate:
        checkpoints = find_all_checkpoints(run_dir)
        if not checkpoints:
            print("Warning: No checkpoints found, skipping visualization regeneration")
        else:
            print(f"Found {len(checkpoints)} checkpoint(s) to process")

            # Clear existing visualization directories ONCE before processing all checkpoints
            val_viz_dir = run_dir / "val_visualizations"
            test_eval_dir = run_dir / "test_evaluations"

            if val_viz_dir.exists():
                print(f"Clearing existing validation visualizations: {val_viz_dir}")
                shutil.rmtree(val_viz_dir)

            if test_eval_dir.exists():
                print(f"Clearing existing test evaluations: {test_eval_dir}")
                shutil.rmtree(test_eval_dir)

            # Process each checkpoint and collect metrics
            for idx, checkpoint_path in enumerate(checkpoints, 1):
                print(
                    f"\n[{idx}/{len(checkpoints)}] Processing checkpoint: {checkpoint_path.name}"
                )

                # Extract epoch from checkpoint filename (use index as fallback)
                checkpoint_epoch = extract_epoch_from_checkpoint(checkpoint_path, idx)
                print(f"  Using epoch: {checkpoint_epoch}")

                success, checkpoint_metrics = regenerate_visualizations(
                    run_dir,
                    config,
                    checkpoint_path,
                    checkpoint_epoch,
                    server_config,
                    high_res,
                    val_viz_n,
                    test_eval_n,
                )
                if success:
                    # Merge metrics from this checkpoint (will override if names are the same)
                    all_checkpoint_metrics.update(checkpoint_metrics)
                else:
                    print(f"  Warning: Failed to process {checkpoint_path.name}")

    # Detect experiment type
    experiment_name = detect_experiment_type(config)
    run_name = run_dir.name

    try:
        # Setup MLflow
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Check if run already exists
        existing_runs = mlflow.search_runs(
            filter_string=f"tags.mlflow.runName = '{run_name}'", output_format="list"
        )

        # Check for duplicate runs (multiple runs with same name)
        if len(existing_runs) > 1:
            print(
                f"❌ Error: Found {len(existing_runs)} runs with name '{run_name}' in experiment '{experiment_name}'"
            )
            print("   Cannot safely update - multiple runs detected in MLflow.")
            print("   Run IDs found:")
            for i, run in enumerate(existing_runs, 1):
                run_id_display = run.info.run_id  # type: ignore[attr-defined]
                print(f"     {i}. {run_id_display}")
            print("\n   To fix this issue:")
            print(f"     1. Go to MLflow UI: {tracking_uri}")
            print(
                "     2. Delete or rename the duplicate runs you don't want to update"
            )
            print(f"     3. Keep only ONE run with name '{run_name}'")
            return False

        # Determine whether to create new run or update existing
        if len(existing_runs) == 1:
            if regenerate:
                # Safe to update - only one run exists
                existing_run_id = existing_runs[0].info.run_id  # type: ignore[attr-defined]
                print(f"Updating existing run: {run_name} (ID: {existing_run_id})")
                run_id = existing_run_id
            else:
                print(
                    f"Skipping {run_dir.name}: Run already exists in MLflow (use --regenerate to update)"
                )
                return False
        else:
            # No existing run - create new one
            print(f"Creating new run: {run_name}")
            run_id = None

        # Start or resume run
        with mlflow.start_run(
            run_id=run_id, run_name=run_name if run_id is None else None
        ) as run:
            if regenerate and run_id is not None:
                # Only log new metrics (don't re-log old ones)
                if all_checkpoint_metrics:
                    print(
                        f"  Logging {len(all_checkpoint_metrics)} new test metrics..."
                    )
                    mlflow.log_metrics(all_checkpoint_metrics)
            else:
                # Initial upload: log all parameters and metrics
                params = (
                    mlflow_utils.extract_mlflow_params(config, {})  # type: ignore[misc]
                    if MLFLOW_AVAILABLE
                    else {}
                )
                mlflow.log_params(params)

                # Merge new metrics with old ones for initial upload
                metrics.update(all_checkpoint_metrics)
                mlflow.log_metrics(metrics)

                # Log checkpoints as artifacts (only on initial upload, not regenerate)
                checkpoint_dir = run_dir / "checkpoints"
                if checkpoint_dir.exists():
                    print("  Uploading checkpoints...")
                    mlflow.log_artifacts(
                        str(checkpoint_dir), artifact_path="checkpoints"
                    )

                # Log config as artifact
                config_path = run_dir / "conf.json"
                if config_path.exists():
                    mlflow.log_artifact(str(config_path), artifact_path="config")

            # Delete old visualization artifacts before uploading new ones
            # (prevents accumulation of stale/duplicate PNGs)
            if run_id:
                delete_mlflow_artifact_directories(
                    run_id, ["val_visualizations", "test_evaluations"]
                )

            # Always upload visualization directories (new or updated)
            val_viz_dir = run_dir / "val_visualizations"
            if val_viz_dir.exists():
                print("  Uploading val_visualizations...")
                mlflow.log_artifacts(
                    str(val_viz_dir), artifact_path="val_visualizations"
                )

            test_eval_dir = run_dir / "test_evaluations"
            if test_eval_dir.exists():
                print("  Uploading test_evaluations...")
                mlflow.log_artifacts(
                    str(test_eval_dir), artifact_path="test_evaluations"
                )

        action = "updated" if run_id is not None else "uploaded"
        print(f"✅ Successfully {action} {run_dir.name} to MLflow")
        print(f"   Experiment: {experiment_name}")
        if regenerate:
            print(f"   New test metrics: {len(all_checkpoint_metrics)} added")
        else:
            print(f"   Metrics: {len(metrics)} logged")
        return True

    except Exception as e:
        print(f"❌ Failed to upload {run_dir.name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def upload_batch_runs(
    output_dir: Path,
    tracking_uri: str,
    server_name: str,
    experiment_type: Optional[str] = None,
    regenerate: bool = False,
    high_res: bool = False,
    val_viz_n: int = 4,
    test_eval_n: int = 4,
) -> int:
    """Upload multiple runs to MLflow."""
    print(f"Scanning for runs in: {output_dir}")

    # Find all run directories
    run_dirs = [
        d for d in output_dir.iterdir() if d.is_dir() and (d / "conf.json").exists()
    ]

    if not run_dirs:
        print("No runs found with configuration files")
        return 0

    # Filter by experiment type if specified
    if experiment_type:
        filtered_runs = []
        for run_dir in run_dirs:
            config = load_run_config(run_dir)
            if config and detect_experiment_type(config) == experiment_type:
                filtered_runs.append(run_dir)
        run_dirs = filtered_runs
        print(f"Found {len(run_dirs)} runs of type '{experiment_type}'")
    else:
        print(f"Found {len(run_dirs)} total runs")

    # Upload each run
    successful_uploads = 0
    for run_dir in sorted(run_dirs):
        if upload_single_run(
            run_dir,
            tracking_uri,
            server_name,
            regenerate,
            high_res,
            val_viz_n,
            test_eval_n,
        ):
            successful_uploads += 1

    print(
        f"\n📊 Upload Summary: {successful_uploads}/{len(run_dirs)} runs uploaded successfully"
    )
    return successful_uploads


def main():
    """Main upload function."""
    parser = argparse.ArgumentParser(
        description="Upload completed runs to MLflow with checkpoint evaluation"
    )
    parser.add_argument("run_dir", nargs="?", help="Single run directory to upload")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory containing runs",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="https://mlflow.developerjose.duckdns.org/",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        help="Filter by experiment type (e.g., baseline_ci, baseline_debris)",
    )
    parser.add_argument(
        "--batch", action="store_true", help="Upload all runs from output directory"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate visualizations for ALL checkpoints and update existing MLflow run with new test metrics",
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Generate high-resolution visualizations for dissertation",
    )
    parser.add_argument(
        "--val-viz-n",
        type=int,
        default=4,
        help="Number of validation visualizations per third (total = 3*n)",
    )
    parser.add_argument(
        "--test-eval-n",
        type=int,
        default=4,
        help="Number of test evaluation visualizations per third (total = 3*n)",
    )
    parser.add_argument(
        "--server",
        type=str,
        required=True,
        choices=["desktop", "bilbo", "frodo"],
        help="Server name for loading image_dir path (must exist in configs/servers.yaml)",
    )

    args = parser.parse_args()

    if not MLFLOW_AVAILABLE:
        print("Error: MLflow utilities not available. Cannot proceed.")
        return 1

    if args.batch:
        # Batch upload mode
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"Error: Output directory {output_dir} does not exist")
            return 1

        successful_uploads = upload_batch_runs(
            output_dir,
            args.tracking_uri,
            args.server,
            args.experiment_type,
            args.regenerate,
            args.high_res,
            args.val_viz_n,
            args.test_eval_n,
        )
        return 0 if successful_uploads > 0 else 1

    elif args.run_dir:
        # Single run upload mode
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: Run directory {run_dir} does not exist")
            return 1

        success = upload_single_run(
            run_dir,
            args.tracking_uri,
            args.server,
            args.regenerate,
            args.high_res,
            args.val_viz_n,
            args.test_eval_n,
        )
        return 0 if success else 1

    else:
        print("Error: Must specify either a run directory or use --batch")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
