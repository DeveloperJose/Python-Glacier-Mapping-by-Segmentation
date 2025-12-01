#!/usr/bin/env python3
"""
Worker script that runs on GPU servers.
Finds and runs pending experiments assigned to this server and GPU.

Usage on server:
  # One-shot (check once and exit) - runs on GPU 0
  uv run python -m glacier_mapping.distributed.worker --server bilbo --gpu 0 --once

  # Loop mode (check every N seconds, default 60s = 1min)
  uv run python -m glacier_mapping.distributed.worker --server bilbo --gpu 0 --loop 60

  # Run workers for both GPUs on bilbo (in separate terminals)
  uv run python -m glacier_mapping.distributed.worker --server bilbo --gpu 0 --loop 60
  uv run python -m glacier_mapping.distributed.worker --server bilbo --gpu 1 --loop 60
"""

import argparse
import gc
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml


def test_experiment_memory(
    exp_config: dict, gpu_rank: int, server_name: str
) -> tuple[bool, str]:
    """
    Test if an experiment can fit in GPU memory.

    Uses Framework.from_config() and fetch_loaders() to simulate actual training:
    - Load model
    - Create data loaders
    - Run one forward + backward + optimizer step

    Returns: (success: bool, message: str)
    """
    # Add glacier_mapping to path if needed
    code_path = Path.cwd()
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))

    from glacier_mapping.data.data import fetch_loaders
    from glacier_mapping.core.frame import Framework

    # Write test config to a temp file
    test_config_path = Path("conf/unet_train_memtest.yaml")
    test_config_path.write_text(
        yaml.dump(exp_config, sort_keys=False, default_flow_style=False)
    )

    try:
        # Set GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_rank)
        torch.cuda.set_device(0)  # After CUDA_VISIBLE_DEVICES, we always use device 0

        print(f"  [Memory Test] Testing on GPU {gpu_rank}...")
        print(f"  [Memory Test] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  [Memory Test] GPU name: {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  [Memory Test] GPU memory: {mem_gb:.2f} GB")

        # Initialize framework (loads model)
        frame = Framework.from_config(test_config_path, device=0)

        # Fetch loaders (simulates data loading)
        train_loader, val_loader, test_loader = fetch_loaders(
            processed_dir=str(exp_config["loader_opts"]["processed_dir"]),
            batch_size=int(exp_config["loader_opts"]["batch_size"]),
            use_channels=list(exp_config["loader_opts"]["use_channels"]),
            output_classes=list(exp_config["loader_opts"]["output_classes"]),
            class_names=list(exp_config["loader_opts"]["class_names"]),
            normalize=str(exp_config["loader_opts"]["normalize"]),
        )

        # Test one training step
        print("  [Memory Test] Running test forward + backward pass...")
        loss_val = 0.0
        for x, y_onehot, y_int in train_loader:
            # Forward + backward (same as training)
            y_hat, loss = frame.optimize(x, y_onehot, y_int.squeeze(-1))
            frame.step()
            loss_val = loss.item()

            # Only test one batch
            break

        print(f"  [Memory Test] ✓ Success! Loss: {loss_val:.4f}")

        # Clean up
        del frame
        del train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

        test_config_path.unlink(missing_ok=True)

        return True, "success"

    except torch.cuda.OutOfMemoryError:
        print("  [Memory Test] ✗ Out of memory")

        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
        test_config_path.unlink(missing_ok=True)

        return False, "out_of_memory"

    except Exception as e:
        print(f"  [Memory Test] ✗ Error: {e}")

        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
        test_config_path.unlink(missing_ok=True)

        return False, f"error: {str(e)}"


def get_experiment_status(
    exp_id: str, server_name: str, gpu_rank: int, run_name: str
) -> str:
    """
    Determine experiment status from filesystem.

    Returns: 'pending' | 'running' | 'completed' | 'failed'
    """
    results_dir = Path("output/runs") / f"{run_name}_{server_name}_gpu{gpu_rank}"

    if not results_dir.exists():
        return "pending"

    if (results_dir / "FAILED").exists():
        return "failed"

    if (results_dir / "checkpoints_summary.json").exists():
        return "completed"

    return "running"


def get_next_experiment(server_name: str, gpu_rank: int):
    """Find next pending experiment for this server and GPU."""
    conf_dir = Path("conf/experiments")

    # Find all experiment YAML files
    exp_files = sorted(conf_dir.glob("exp_*.yaml"))

    for exp_file in exp_files:
        exp_id = exp_file.stem
        exp_config = yaml.safe_load(exp_file.read_text())

        # Check if assigned to this server and GPU
        if exp_config.get("server") != server_name:
            continue

        if exp_config.get("gpu_rank") != gpu_rank:
            continue

        # Check status
        run_name = exp_config["training_opts"]["run_name"]
        status = get_experiment_status(exp_id, server_name, gpu_rank, run_name)

        if status == "pending":
            return exp_id, exp_file, exp_config

    return None, None, None


def run_experiment(
    server_name: str, gpu_rank: int, exp_id: str, exp_file: Path, exp_config: dict
) -> bool:
    """Execute the training experiment."""
    servers_cfg = yaml.safe_load(Path("conf/servers.yaml").read_text())
    server = servers_cfg[server_name]

    # Update paths for this server - use consolidated output structure
    # The config already has correct paths from submit.py, just use as-is
    # No need to modify paths since submit.py already set them correctly

    print(f"\n[{exp_id}] Starting training on {server_name} GPU {gpu_rank}...")
    print(f"[{exp_id}] Output: {exp_config['training_opts']['output_dir']}")

    # Create marker to show running status
    run_name = exp_config["training_opts"]["run_name"]
    results_marker = Path("output/runs") / f"{run_name}_{server_name}_gpu{gpu_rank}"
    results_marker.mkdir(parents=True, exist_ok=True)
    (results_marker / "RUNNING").write_text(f"Started: {datetime.now().isoformat()}\n")

    # Run training with CUDA_VISIBLE_DEVICES set
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_rank)

        # Convert experiment config path to be relative to code_path
        # exp_file is relative to glacier_mapping/, need to make it relative to code_path
        exp_file_from_root = Path("glacier_mapping") / exp_file

        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "glacier_mapping.scripts.train",
                "--config",
                str(exp_file_from_root),
            ],
            cwd=server["code_path"],
            env=env,
            check=True,
            capture_output=False,
        )
        print(f"\n[{exp_id}] ✓ Training completed successfully!")

        # Remove running marker
        (results_marker / "RUNNING").unlink(missing_ok=True)

        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[{exp_id}] ✗ Training failed with exit code {e.returncode}")

        # Mark as failed
        (results_marker / "RUNNING").unlink(missing_ok=True)
        (results_marker / "FAILED").write_text(
            f"Failed: {datetime.now().isoformat()}\nExit code: {e.returncode}\n"
        )

        return False


def git_pull() -> None:
    """Pull latest changes from git."""
    try:
        subprocess.run(
            ["git", "pull", "origin", "master"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✓ Git pull successful")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Git pull failed: {e.stderr}")


def worker_loop(server_name: str, gpu_rank: int, interval: int | None = None) -> None:
    """
    Main worker loop.

    Checks for pending experiments assigned to this server + GPU.
    Tests memory before running, waits and retries on OOM.
    """
    print(f"\n{'=' * 80}")
    print(f"WORKER STARTING: {server_name} GPU {gpu_rank}")
    print(f"Mode: {'one-shot' if interval is None else f'loop (every {interval}s)'}")
    print(f"{'=' * 80}\n")

    # Track retry counts for OOM experiments
    oom_retry_counts = {}
    max_retries = 5
    retry_wait_seconds = 60

    while True:
        # Git pull to get latest experiments
        git_pull()

        # Get next experiment
        exp_id, exp_file, exp_config = get_next_experiment(server_name, gpu_rank)

        if exp_id is not None and exp_file is not None and exp_config is not None:
            print(f"\n{'=' * 80}")
            print(f"Found experiment: {exp_id}")
            print(f"Config: {exp_file}")
            print(f"Server: {server_name} GPU: {gpu_rank}")
            print(f"{'=' * 80}")

            # Test memory first
            print(f"\n[{exp_id}] Testing GPU memory before launch...")
            can_run, message = test_experiment_memory(exp_config, gpu_rank, server_name)

            if can_run:
                print(f"[{exp_id}] Memory test passed, launching training...")

                # Reset retry count on success
                oom_retry_counts.pop(exp_id, None)

                # Run experiment
                success = run_experiment(
                    server_name, gpu_rank, exp_id, exp_file, exp_config
                )

                if success:
                    print(f"[{exp_id}] ✓ Training completed successfully!")
                    print(
                        f"[{exp_id}] Results available at: {exp_config['training_opts']['output_dir']}"
                    )
                    print(
                        f"[{exp_id}] Use monitor.py --sync-all to pull results to desktop"
                    )
            else:
                # OOM or error
                if message == "out_of_memory":
                    # Track retry count
                    retries = oom_retry_counts.get(exp_id, 0)
                    oom_retry_counts[exp_id] = retries + 1

                    if retries < max_retries:
                        print(
                            f"[{exp_id}] Out of memory (retry {retries + 1}/{max_retries})"
                        )
                        print(
                            f"[{exp_id}] Waiting {retry_wait_seconds}s before retry..."
                        )
                        time.sleep(retry_wait_seconds)
                        continue  # Try again immediately
                    else:
                        print(
                            f"[{exp_id}] Out of memory after {max_retries} retries, skipping"
                        )
                        # Mark as failed
                        run_name = exp_config["training_opts"]["run_name"]
                        results_marker = (
                            Path("output/runs")
                            / f"{run_name}_{server_name}_gpu{gpu_rank}"
                        )
                        results_marker.mkdir(parents=True, exist_ok=True)
                        (results_marker / "FAILED").write_text(
                            f"Failed: {datetime.now().isoformat()}\n"
                            f"Reason: Out of memory after {max_retries} retries\n"
                        )
                        oom_retry_counts.pop(exp_id, None)
                else:
                    print(f"[{exp_id}] Memory test failed: {message}")
                    # Mark as failed
                    run_name = exp_config["training_opts"]["run_name"]
                    results_marker = (
                        Path("output/runs") / f"{run_name}_{server_name}_gpu{gpu_rank}"
                    )
                    results_marker.mkdir(parents=True, exist_ok=True)
                    (results_marker / "FAILED").write_text(
                        f"Failed: {datetime.now().isoformat()}\n"
                        f"Reason: Memory test failed - {message}\n"
                    )
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{timestamp}] No pending experiments for {server_name} GPU {gpu_rank}"
            )

        if interval is None:
            break  # One-shot mode

        print(f"\nSleeping for {interval}s...\n")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed experiment worker")
    parser.add_argument(
        "--server",
        required=True,
        choices=["desktop", "bilbo", "frodo"],
        help="Server name (must match conf/servers.yaml)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="GPU rank to use (0 or 1 for bilbo, 0 for desktop)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (default: loop)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=60,
        help="Loop interval in seconds (default: 60)",
    )
    args = parser.parse_args()

    interval = None if args.once else args.loop
    worker_loop(args.server, args.gpu, interval)
