#!/usr/bin/env python3
"""
Monitor experiment status across all servers.

Usage:
  uv run python -m glacier_mapping.distributed.monitor
  uv run python -m glacier_mapping.distributed.monitor --detailed
  uv run python -m glacier_mapping.distributed.monitor --server bilbo
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Conservative line width for all terminals
LINE_WIDTH = 60


def get_experiment_status(
    exp_id: str, server_name: str, gpu_rank: int, run_name: str, debug: bool = False
) -> str:
    """
    Determine experiment status from filesystem.

    Returns: 'pending' | 'running' | 'completed' | 'failed'
    """
    # For local experiments, check local filesystem
    if server_name == "desktop":
        results_dir = Path("output/runs") / f"{run_name}_{server_name}_gpu{gpu_rank}"
    else:
        # For remote experiments, we'll check via SSH in get_remote_experiment_status
        # Pass debug flag from monitor_experiments
        debug = locals().get("debug", False)  # Get debug from outer scope
        return get_remote_experiment_status(
            server_name, exp_id, gpu_rank, run_name, debug
        )

    if not results_dir.exists():
        return "pending"

    if (results_dir / "FAILED").exists():
        return "failed"

    if (results_dir / "checkpoints_summary.json").exists():
        return "completed"

    return "running"


def get_remote_experiment_status(
    server_name: str, exp_id: str, gpu_rank: int, run_name: str, debug: bool = False
) -> str:
    """Check experiment status on remote server via SSH."""
    servers_cfg = yaml.safe_load(Path("conf/servers.yaml").read_text())
    server = servers_cfg[server_name]

    if server_name == "desktop":
        return get_experiment_status(exp_id, server_name, gpu_rank, run_name)

    # Check remote filesystem via SSH with debugging
    remote_results_dir = (
        f"{server['code_path']}/output/runs/{run_name}_{server_name}_gpu{gpu_rank}"
    )

    if debug:
        print(f"DEBUG: Checking {server_name} at {remote_results_dir}")

    # First, test if directory exists and list contents
    ssh_cmd = [
        "ssh",
        server["ssh_host"],
        f"ls -la {remote_results_dir} 2>/dev/null || echo 'DIRECTORY_MISSING'",
    ]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
        if "DIRECTORY_MISSING" in result.stdout:
            if debug:
                print(f"DEBUG: {server_name} directory missing - status: pending")
            return "pending"
        if debug:
            print(f"DEBUG: {server_name} directory contents:\n{result.stdout}")
    except subprocess.TimeoutExpired:
        if debug:
            print(f"DEBUG: {server_name} SSH timeout - status: pending")
        return "pending"
    except Exception as e:
        if debug:
            print(f"DEBUG: {server_name} SSH error: {e} - status: pending")
        return "pending"

    # Check for FAILED marker
    ssh_cmd = [
        "ssh",
        server["ssh_host"],
        f"test -f {remote_results_dir}/FAILED && echo 'FAILED' || echo 'NO_FAILED'",
    ]

    if debug:
        print(f"DEBUG: {server_name} FAILED check SSH command: {' '.join(ssh_cmd)}")

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=5)
        if debug:
            print(f"DEBUG: {server_name} FAILED check return code: {result.returncode}")
            print(
                f"DEBUG: {server_name} FAILED check stdout: '{result.stdout.strip()}'"
            )
            print(f"DEBUG: {server_name} FAILED check stderr: '{result.stderr}'")

        if result.stdout.strip() == "FAILED":
            if debug:
                print(f"DEBUG: {server_name} has FAILED marker - status: failed")
            return "failed"
    except Exception as e:
        if debug:
            print(f"DEBUG: {server_name} FAILED check error: {e}")

    # Check for completion marker
    ssh_cmd = [
        "ssh",
        server["ssh_host"],
        f"test -f {remote_results_dir}/checkpoints_summary.json && echo 'COMPLETED' || echo 'RUNNING'",
    ]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=5)
        status = "completed" if result.stdout.strip() == "COMPLETED" else "running"
        if debug:
            print(
                f"DEBUG: {server_name} completion check: {result.stdout.strip()} -> status: {status}"
            )
        return status
    except Exception as e:
        if debug:
            print(f"DEBUG: {server_name} completion check error: {e}")
        return "running"


def is_synced_locally(
    exp_id: str, server_name: str, gpu_rank: int, run_name: str
) -> bool:
    """Check if experiment results are already synced locally."""
    local_path = Path("output/runs") / f"{run_name}_{server_name}_gpu{gpu_rank}"
    if not local_path.exists():
        return False

    # Check for key files to verify complete sync
    required_files = ["conf.json", "checkpoints_summary.json"]
    return all((local_path / f).exists() for f in required_files)


def sync_experiment_results(
    server_name: str, exp_id: str, gpu_rank: int, run_name: str
) -> bool:
    """Pull experiment results from remote server to desktop."""
    if server_name == "desktop":
        print(f"[{exp_id}] Already on desktop - no sync needed")
        return True

    servers_cfg = yaml.safe_load(Path("conf/servers.yaml").read_text())
    server = servers_cfg[server_name]

    source = f"{server['ssh_host']}:{server['code_path']}/output/runs/{run_name}_{server_name}_gpu{gpu_rank}/"
    dest = Path("output/runs") / f"{run_name}_{server_name}_gpu{gpu_rank}"

    print(f"[{exp_id}] Syncing from {server_name}...")
    print(f"  Source: {source}")
    print(f"  Dest:   {dest}")

    # Create local directory
    dest.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["rsync", "-avz", "--partial", "--progress", source, str(dest) + "/"],
            check=True,
        )
        print(f"[{exp_id}] ✓ Sync completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{exp_id}] ✗ Sync failed: {e}")
        return False


def sync_all_completed_experiments(debug: bool = False) -> None:
    """Sync all completed experiments from all servers."""
    servers_cfg = yaml.safe_load(Path("conf/servers.yaml").read_text())
    conf_dir = Path("conf/experiments")

    # Fixed-width formatting
    line = "=" * LINE_WIDTH
    print(f"\n{line}")
    print("SYNCING ALL COMPLETED EXPERIMENTS")
    print(line + "\n")

    exp_files = sorted(conf_dir.glob("exp_*.yaml"))

    if not exp_files:
        print("No experiments found in conf/experiments/")
        return

    synced_count = 0
    failed_count = 0

    for exp_file in exp_files:
        exp_id = exp_file.stem
        exp_config = yaml.safe_load(exp_file.read_text())

        server = exp_config.get("server", "unknown")
        gpu_rank = exp_config.get("gpu_rank", 0)
        run_name = exp_config.get("training_opts", {}).get("run_name", "unknown")

        # Check if experiment is completed
        status = get_experiment_status(exp_id, server, gpu_rank, run_name, debug)
        if status != "completed":
            continue

        # Check if already synced
        if is_synced_locally(exp_id, server, gpu_rank, run_name):
            print(f"[{exp_id}] ✓ Already synced")
            synced_count += 1
        else:
            # Sync experiment
            if sync_experiment_results(server, exp_id, gpu_rank, run_name):
                synced_count += 1
            else:
                failed_count += 1

    # Fixed-width summary
    print(f"\n{line}")
    print(f"Sync Summary: {synced_count} synced, {failed_count} failed")
    print(line + "\n")

    exp_files = sorted(conf_dir.glob("exp_*.yaml"))
    synced_count = 0
    failed_count = 0

    for exp_file in exp_files:
        exp_id = exp_file.stem
        exp_config = yaml.safe_load(exp_file.read_text())

        server = exp_config.get("server", "unknown")
        gpu_rank = exp_config.get("gpu_rank", 0)
        run_name = exp_config.get("training_opts", {}).get("run_name", "unknown")

        # Check if experiment is completed
        status = get_experiment_status(exp_id, server, gpu_rank, run_name, debug)
        if status != "completed":
            continue

        # Check if already synced
        if is_synced_locally(exp_id, server, gpu_rank, run_name):
            print(f"[{exp_id}] ✓ Already synced")
            synced_count += 1
            continue

        # Sync the experiment
        if sync_experiment_results(server, exp_id, gpu_rank, run_name):
            synced_count += 1
        else:
            failed_count += 1

    print("\n" + "=" * 80)
    print(f"Sync Summary: {synced_count} synced, {failed_count} failed")
    print("=" * 80 + "\n")


def monitor_experiments(
    detailed: bool = False, server_filter: str | None = None, debug: bool = False
) -> None:
    """Display experiment status."""
    conf_dir = Path("conf/experiments")

    # Find all experiment YAML files
    exp_files = sorted(conf_dir.glob("exp_*.yaml"))

    if not exp_files:
        print("No experiments found in conf/experiments/")
        print("\nTo create an experiment:")
        print(
            "  uv run python -m glacier_mapping.distributed.submit --server bilbo --gpu 0"
        )
        return

    # Fixed-width formatting
    line = "=" * LINE_WIDTH
    print(f"\n{line}")
    print("EXPERIMENT STATUS")
    print(line + "\n")

    # Group by status
    status_groups: dict[str, list[tuple[str, str, dict, str, str]]] = {
        "pending": [],
        "running": [],
        "completed": [],
        "failed": [],
    }

    for exp_file in exp_files:
        exp_id = exp_file.stem
        exp_config = yaml.safe_load(exp_file.read_text())

        server = exp_config.get("server", "unknown")
        gpu_rank = exp_config.get("gpu_rank", 0)

        # Apply server filter if specified
        if server_filter and server != server_filter:
            continue

        run_name = exp_config.get("training_opts", {}).get("run_name", "unknown")
        status = get_experiment_status(exp_id, server, gpu_rank, run_name, debug)
        status_groups[status].append(
            (exp_id, server, exp_config, exp_file.name, run_name)
        )

    # Display each status group
    for status_name, exps in status_groups.items():
        if not exps:
            continue

        sep = "-" * LINE_WIDTH
        print(f"\n{status_name.upper()} ({len(exps)}):")
        print(sep)

        for exp_id, server, exp_config, filename, run_name in exps:
            gpu = exp_config.get("gpu_rank", "?")
            run_name = exp_config.get("training_opts", {}).get("run_name", "?")

            # Simple fixed-width formatting
            line = f"  {exp_id:<12} {server:<6} GPU:{gpu} {run_name[:10]:<10}"

            # Add sync status for completed experiments
            if status_name == "completed":
                if is_synced_locally(exp_id, server, gpu, run_name):
                    line += " ✓SYNC"
                else:
                    line += " ✗SYNC"

            if detailed:
                line += f"\n    File: experiments/conf/{filename}"
                processed_dir = exp_config.get("loader_opts", {}).get(
                    "processed_dir", "?"
                )
                line += f"\n    Dataset: {processed_dir}"
                epochs = exp_config.get("training_opts", {}).get("epochs", "?")
                line += f"\n    Epochs: {epochs}"

                # Show results path if exists
                results_path = Path("output/runs") / f"{run_name}_{server}_gpu{gpu}"
                if results_path.exists():
                    line += f"\n    Results: {results_path}"

            print(line)

    # Summary
    total = sum(len(exps) for exps in status_groups.values())

    # Fixed-width separator
    sep = "=" * LINE_WIDTH
    print(f"\n{sep}")
    print(f"Total experiments: {total}")
    if server_filter:
        print(f"Filtered by server: {server_filter}")

    # Show status counts
    counts = {status: len(exps) for status, exps in status_groups.items() if exps}
    if counts:
        print(
            f"Status: {', '.join(f'{status}={count}' for status, count in counts.items())}"
        )

    print(sep + "\n")


def interactive_monitor(server_filter: str | None = None, debug: bool = False) -> None:
    """Interactive monitor with keypress controls."""
    import select
    import termios
    import tty

    def get_key():
        """Get a single keypress from stdin."""
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    # Fixed-width formatting
    line = "=" * LINE_WIDTH
    print(f"\n{line}")
    print("INTERACTIVE MONITOR MODE")
    print(line)
    print("Controls: [s]ync all, [r]efresh, [q]uit")
    print(line + "\n")

    # Set up terminal for non-blocking input
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())

        while True:
            # Display current status
            monitor_experiments(
                detailed=False, server_filter=server_filter, debug=debug
            )

            print(f"\n{line}")
            print("Press: [s]ync all, [r]efresh, [q]uit")
            print(line)

            # Wait for keypress with timeout
            key = None
            start_time = time.time()
            while time.time() - start_time < 5:  # 5-second timeout
                key = get_key()
                if key:
                    break
                time.sleep(0.1)

            if key:
                key = key.lower()
                if key == "q":
                    print("\n👋 Exiting interactive monitor")
                    break
                elif key == "s":
                    print("\n🔄 Syncing all completed experiments...")
                    sync_all_completed_experiments(debug)
                elif key == "r":
                    print("\n🔄 Refreshing...")
                    continue
                else:
                    print(f"\n❓ Unknown key: {key}")
            else:
                # Timeout - auto refresh
                print("\n⏰ Auto-refreshing...")

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor distributed experiments")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information",
    )
    parser.add_argument(
        "--server",
        choices=["desktop", "bilbo"],
        help="Filter by server",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode with keypress controls",
    )
    parser.add_argument(
        "--sync-all",
        action="store_true",
        help="Sync all completed experiments from remote servers",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for remote status checking",
    )
    args = parser.parse_args()

    if args.sync_all:
        sync_all_completed_experiments(args.debug)
    elif args.interactive:
        interactive_monitor(args.server)
    else:
        monitor_experiments(args.detailed, args.server, args.debug)
