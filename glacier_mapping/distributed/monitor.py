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

# Conservative line width for better compatibility
LINE_WIDTH = 80

# Status cache to avoid redundant SSH calls
_status_cache: dict[str, tuple[str, float]] = {}  # key -> (status, timestamp)
_cache_ttl = 15  # seconds


def get_experiment_status(
    exp_id: str,
    server_name: str,
    gpu_rank: int,
    run_name: str,
    debug: bool = False,
    use_cache: bool = True,
) -> str:
    """
    Determine experiment status from filesystem.

    Returns: 'pending' | 'running' | 'completed' | 'failed'
    """
    # Check cache first (if enabled and not local)
    if use_cache and server_name != "desktop":
        cache_key = f"{exp_id}_{server_name}_{gpu_rank}"
        if cache_key in _status_cache:
            cached_status, cached_time = _status_cache[cache_key]
            age = time.time() - cached_time
            if age < _cache_ttl:
                if debug:
                    print(f"DEBUG: Using cached status for {exp_id} (age: {age:.1f}s)")
                return cached_status

    # For local experiments, check local filesystem
    if server_name == "desktop":
        # Load servers config to get output path
        servers_cfg = yaml.safe_load(Path("conf/servers.yaml").read_text())
        server_cfg = servers_cfg[server_name]
        results_dir = (
            Path(server_cfg["output_path"])
            / "runs"
            / f"{run_name}_{server_name}_gpu{gpu_rank}"
        )

        if not results_dir.exists():
            return "pending"

        if (results_dir / "FAILED").exists():
            return "failed"

        if (results_dir / "checkpoints_summary.json").exists():
            return "completed"

        return "running"
    else:
        # For remote experiments, we'll check via SSH in get_remote_experiment_status
        status = get_remote_experiment_status(
            server_name, exp_id, gpu_rank, run_name, debug
        )

        # Cache the result
        if use_cache:
            cache_key = f"{exp_id}_{server_name}_{gpu_rank}"
            _status_cache[cache_key] = (status, time.time())

        return status


def get_remote_experiment_status(
    server_name: str, exp_id: str, gpu_rank: int, run_name: str, debug: bool = False
) -> str:
    """Check experiment status on remote server via SSH (optimized - single call)."""
    servers_cfg = yaml.safe_load(Path("conf/servers.yaml").read_text())
    server = servers_cfg[server_name]

    if server_name == "desktop":
        return get_experiment_status(exp_id, server_name, gpu_rank, run_name)

    # Check remote filesystem via SSH with debugging
    remote_results_dir = (
        f"{server['output_path']}/runs/{run_name}_{server_name}_gpu{gpu_rank}"
    )

    if debug:
        print(f"DEBUG: Checking {server_name} at {remote_results_dir}")

    # Combine all checks into a single SSH call for efficiency
    ssh_cmd = [
        "ssh",
        server["ssh_host"],
        f"""
        if [ ! -d {remote_results_dir} ]; then
            echo "DIRECTORY_MISSING"
        elif [ -f {remote_results_dir}/FAILED ]; then
            echo "FAILED"
        elif [ -f {remote_results_dir}/checkpoints_summary.json ]; then
            echo "COMPLETED"
        else
            echo "RUNNING"
        fi
        """,
    ]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
        status_output = result.stdout.strip()

        if debug:
            print(f"DEBUG: {server_name} status output: '{status_output}'")

        # Map output to status
        if status_output == "DIRECTORY_MISSING":
            return "pending"
        elif status_output == "FAILED":
            return "failed"
        elif status_output == "COMPLETED":
            return "completed"
        elif status_output == "RUNNING":
            return "running"
        else:
            if debug:
                print(f"DEBUG: {server_name} unknown status: {status_output}")
            return "pending"

    except subprocess.TimeoutExpired:
        if debug:
            print(f"DEBUG: {server_name} SSH timeout - status: pending")
        return "pending"
    except Exception as e:
        if debug:
            print(f"DEBUG: {server_name} SSH error: {e} - status: pending")
        return "pending"


def is_synced_locally(
    exp_id: str, server_name: str, gpu_rank: int, run_name: str, status: str = "unknown"
) -> bool:
    """Check if experiment results are already synced locally."""
    local_path = Path("output/runs") / f"{run_name}_{server_name}_gpu{gpu_rank}"
    if not local_path.exists():
        return False

    # For completed experiments, require checkpoints_summary.json
    if status == "completed":
        required_files = ["conf.json", "checkpoints_summary.json"]
        return all((local_path / f).exists() for f in required_files)

    # For running experiments, just check for conf.json (lighter check)
    return (local_path / "conf.json").exists()


def get_sync_age(server_name: str, gpu_rank: int, run_name: str) -> str:
    """Get age of local sync in human-readable format."""
    local_path = Path("output/runs") / f"{run_name}_{server_name}_gpu{gpu_rank}"
    if not local_path.exists():
        return ""

    # Check modification time of conf.json as proxy for sync time
    conf_file = local_path / "conf.json"
    if not conf_file.exists():
        return ""

    import datetime

    mod_time = datetime.datetime.fromtimestamp(conf_file.stat().st_mtime)
    now = datetime.datetime.now()
    age = now - mod_time

    # Format age nicely
    if age.total_seconds() < 60:
        return "just now"
    elif age.total_seconds() < 3600:
        mins = int(age.total_seconds() / 60)
        return f"{mins}m ago"
    elif age.total_seconds() < 86400:
        hours = int(age.total_seconds() / 3600)
        return f"{hours}h ago"
    else:
        days = int(age.total_seconds() / 86400)
        return f"{days}d ago"


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


def sync_all_experiments(debug: bool = False, include_running: bool = True) -> None:
    """Sync experiments from all servers (optionally including running experiments)."""
    conf_dir = Path("conf/experiments")

    # Fixed-width formatting
    line = "=" * LINE_WIDTH
    print(f"\n{line}")
    if include_running:
        print("SYNCING ALL EXPERIMENTS (INCLUDING RUNNING)")
    else:
        print("SYNCING COMPLETED EXPERIMENTS")
    print(line)
    print()

    exp_files = sorted(conf_dir.glob("exp_*.yaml"))

    if not exp_files:
        print("No experiments found in conf/experiments/")
        return

    synced_count = 0
    failed_count = 0
    skipped_count = 0

    for exp_file in exp_files:
        exp_id = exp_file.stem
        exp_config = yaml.safe_load(exp_file.read_text())

        server = exp_config.get("server", "unknown")
        gpu_rank = exp_config.get("gpu_rank", 0)
        run_name = exp_config.get("training_opts", {}).get("run_name", "unknown")

        # Check experiment status
        status = get_experiment_status(exp_id, server, gpu_rank, run_name, debug)

        # Skip if not completed and we're not including running
        if not include_running and status != "completed":
            skipped_count += 1
            continue

        # Skip pending experiments (nothing to sync yet)
        if status == "pending":
            skipped_count += 1
            continue

        # Check if already synced
        if is_synced_locally(exp_id, server, gpu_rank, run_name, status):
            sync_age = get_sync_age(server, gpu_rank, run_name)
            print(f"[{exp_id}] ✓ Already synced ({sync_age})")
            synced_count += 1
        else:
            # Sync experiment
            if sync_experiment_results(server, exp_id, gpu_rank, run_name):
                synced_count += 1
            else:
                failed_count += 1

    # Summary
    print(f"\n{line}")
    print(
        f"Sync Summary: {synced_count} synced, {failed_count} failed, {skipped_count} skipped"
    )
    print(line)
    print()


def sync_all_completed_experiments(debug: bool = False) -> None:
    """Sync all completed experiments from all servers."""
    sync_all_experiments(debug, include_running=False)


def monitor_experiments(
    detailed: bool = False,
    server_filter: str | None = None,
    debug: bool = False,
    show_progress: bool = False,
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
    print(line)
    print()

    # Group by status
    status_groups: dict[str, list[tuple[str, str, dict, str, str]]] = {
        "pending": [],
        "running": [],
        "completed": [],
        "failed": [],
    }

    # Collect unique remote servers to show progress
    remote_servers = set()
    for exp_file in exp_files:
        exp_config = yaml.safe_load(exp_file.read_text())
        server = exp_config.get("server", "unknown")
        if server != "desktop" and (not server_filter or server == server_filter):
            remote_servers.add(server)

    if show_progress and remote_servers:
        print(f"Checking remote servers: {', '.join(sorted(remote_servers))}...")
        print()

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

            # Compact fixed-width formatting
            line = f"  {exp_id:<18} {server:<7} GPU:{gpu:<1} {run_name[:15]:<15}"

            # Add sync status for all experiments (not just completed)
            if is_synced_locally(exp_id, server, gpu, run_name, status_name):
                sync_age = get_sync_age(server, gpu, run_name)
                if sync_age:
                    line += f" ✓SYNC({sync_age})"
                else:
                    line += " ✓SYNC"
            else:
                if server != "desktop":  # Only show missing sync for remote experiments
                    line += " ✗SYNC"

            if detailed:
                line += f"\n    File: conf/experiments/{filename}"
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

    print(sep)
    print()


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

    def clear_screen():
        """Clear screen and move cursor to top-left."""
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    def print_raw(text=""):
        """Print in raw mode with proper line endings."""
        sys.stdout.write(text + "\r\n")
        sys.stdout.flush()

    # Fixed-width formatting
    line = "=" * LINE_WIDTH

    # Set up terminal for non-blocking input
    old_settings = termios.tcgetattr(sys.stdin)
    paused = False
    refresh_interval = 20  # seconds

    try:
        tty.setraw(sys.stdin.fileno())

        while True:
            # Clear screen and display current status
            clear_screen()

            # Show header with timestamp
            import datetime

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            print_raw(line)
            print_raw(f"INTERACTIVE MONITOR MODE - Last updated: {timestamp}")
            if paused:
                print_raw("STATUS: PAUSED (press [p] to resume)")
            print_raw(line)
            print_raw(
                "Controls: [s]ync completed, [S]ync all, [p]ause, [r]efresh, [q]uit"
            )
            print_raw()

            # Display experiment status - capture and reformat output
            import io
            from contextlib import redirect_stdout

            # Temporarily restore terminal to capture monitor_experiments output
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                monitor_experiments(
                    detailed=False,
                    server_filter=server_filter,
                    debug=debug,
                    show_progress=True,
                )

            # Set back to raw mode
            tty.setraw(sys.stdin.fileno())

            # Print captured output with proper line endings
            for line_text in buffer.getvalue().splitlines():
                print_raw(line_text)

            print_raw()
            print_raw(line)
            if paused:
                print_raw(
                    "Press: [s]ync completed, [S]ync all, [p]ause/resume, [r]efresh, [q]uit"
                )
            else:
                print_raw(
                    f"Press: [s]ync completed, [S]ync all, [p]ause, [r]efresh, [q]uit (auto-refresh: {refresh_interval}s)"
                )
            print_raw(line)

            # Wait for keypress with timeout
            key = None
            start_time = time.time()
            timeout = (
                999999 if paused else refresh_interval
            )  # Very long timeout if paused

            while time.time() - start_time < timeout:
                key = get_key()
                if key:
                    break
                time.sleep(0.1)

            if key:
                key_lower = key.lower()
                if key_lower == "q":
                    # Restore terminal before exiting
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    clear_screen()
                    print("👋 Exiting interactive monitor")
                    return
                elif key == "s":
                    # Restore terminal for sync operations
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    clear_screen()
                    print("🔄 Syncing all completed experiments...")
                    sync_all_completed_experiments(debug)
                    input("Press Enter to continue...")
                    tty.setraw(sys.stdin.fileno())
                elif key == "S":  # Capital S - sync all including running
                    # Restore terminal for sync operations
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    clear_screen()
                    print("🔄 Syncing ALL experiments (including running)...")
                    sync_all_experiments(debug)
                    input("Press Enter to continue...")
                    tty.setraw(sys.stdin.fileno())
                elif key_lower == "p":
                    paused = not paused
                    continue
                elif key_lower == "r":
                    continue
                else:
                    clear_screen()
                    print_raw(f"❓ Unknown key: {key}")
                    time.sleep(1)
            else:
                # Timeout - auto refresh (only if not paused)
                if not paused:
                    continue

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
        choices=["desktop", "bilbo", "frodo"],
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
        interactive_monitor(args.server, args.debug)
    else:
        monitor_experiments(args.detailed, args.server, args.debug)
