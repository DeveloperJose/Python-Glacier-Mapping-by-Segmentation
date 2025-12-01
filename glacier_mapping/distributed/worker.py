#!/usr/bin/env python3
"""
Interactive worker script that runs on GPU servers.
Finds and runs pending experiments assigned to this server and GPU with full interactive control.

Usage on server:
  # Interactive mode (default)
  uv run python -m glacier_mapping.distributed.worker --server bilbo --gpu 0

  # Legacy modes (still supported)
  uv run python -m glacier_mapping.distributed.worker --server bilbo --gpu 0 --once
  uv run python -m glacier_mapping.distributed.worker --server bilbo --gpu 0 --loop 60
"""

import argparse
import contextlib
import gc
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

# Configuration constants
OOM_RETRY_INTERVAL_MINUTES = 30  # Check OOM every 30 minutes
OOM_TIMEOUT_HOURS = 12  # Fail OOM after 12 hours
ERROR_RETRY_INTERVAL_MINUTES = 60  # Check all other errors every 1 hour
ERROR_TIMEOUT_HOURS = 12  # Fail all other errors after 12 hours
OTHER_MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 60
PROCESS_CHECK_INTERVAL = 5  # seconds
REFRESH_INTERVAL = 10  # seconds
CONFIRMATION_TIMEOUT = 30  # seconds
LINE_WIDTH = 80
EXPERIMENT_START_DELAY_SECONDS = 300  # 5 minutes between experiment starts


class ExperimentState:
    """Track state of a single experiment."""

    def __init__(self, exp_id: str):
        self.exp_id = exp_id
        self.process: Optional[subprocess.Popen] = None
        self.status = "pending"  # pending, running, stopped, completed, failed, oom_waiting, error_waiting
        self.start_time: Optional[datetime] = None
        self.stop_time: Optional[datetime] = None
        self.retry_count = 0
        self.oom_wait_start: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.last_error_time: Optional[datetime] = None
        self.pid: Optional[int] = None
        self.gpu_memory_gb = 0.0
        self.current_epoch = 0
        self.current_loss = 0.0
        self.log_file: Optional[Path] = None
        # New fields for universal error handling
        self.error_type: Optional[str] = (
            None  # "oom", "cuda", "data", "import", "config", "unknown"
        )
        self.is_recoverable: bool = False
        self.error_wait_start: Optional[datetime] = None
        self.last_log_pos = 0
        # Track user-initiated stops
        self.user_stopped = False


class InteractiveWorker:
    """Interactive worker with concurrent experiment execution."""

    def __init__(self, server_name: str, gpu_rank: int):
        self.server_name = server_name
        self.gpu_rank = gpu_rank
        self.experiments: Dict[str, ExperimentState] = {}
        self.selected_exp_id: Optional[str] = None
        self.worker_paused = False
        self.running = True
        self.display_needs_refresh = True
        self.last_key_time = 0
        self.total_gpu_memory_gb = 0.0
        self.last_experiment_start_time: Optional[datetime] = None

        # VALIDATE: Must be run from glacier_mapping/ directory
        if not Path("conf/servers.yaml").exists():
            print("\n" + "=" * 80)
            print("ERROR: Worker must be run from glacier_mapping/ directory")
            print("=" * 80)
            print("\nExpected usage:")
            print("  cd glacier_mapping")
            print(
                "  uv run python -m glacier_mapping.distributed.worker --server <name> --gpu <N>"
            )
            print("\nCurrent directory:", Path.cwd())
            print("=" * 80 + "\n")
            sys.exit(1)

        # Load server configuration
        self.servers_cfg = yaml.safe_load(Path("conf/servers.yaml").read_text())
        self.server = self.servers_cfg[server_name]

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Initialize total GPU memory
        self._get_total_gpu_memory()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.running = False
        self._cleanup_all_processes()
        sys.exit(0)

    def _cleanup_all_processes(self):
        """Clean up all running processes."""
        for exp_state in self.experiments.values():
            if exp_state.process and exp_state.process.poll() is None:
                try:
                    exp_state.process.terminate()
                    exp_state.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    exp_state.process.kill()

    def get_key(self) -> Optional[str]:
        """Get keypress"""
        import select

        if sys.stdin not in select.select([sys.stdin], [], [], 0.1)[0]:
            return None

        ch = sys.stdin.read(1)
        return ch

    def clear_screen(self):
        """Clear screen and move cursor to top-left."""
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    def print_raw(self, text: str = ""):
        """Print in raw mode with proper line endings."""
        sys.stdout.write(text + "\r\n")
        sys.stdout.flush()

    def load_experiments(self):
        """Load all experiments from config files."""
        conf_dir = Path("conf/experiments")
        exp_files = sorted(conf_dir.glob("exp_*.yaml"))

        for exp_file in exp_files:
            exp_id = exp_file.stem
            if exp_id not in self.experiments:
                exp_config = yaml.safe_load(exp_file.read_text())

                # Check if assigned to this server and GPU
                if (
                    exp_config.get("server") == self.server_name
                    and exp_config.get("gpu_rank") == self.gpu_rank
                ):
                    exp_state = ExperimentState(exp_id)
                    exp_state.status = self._get_experiment_status(exp_id, exp_config)
                    self.experiments[exp_id] = exp_state

    def _get_experiment_status(self, exp_id: str, exp_config: dict) -> str:
        """Determine experiment status from filesystem."""
        run_name = exp_config["training_opts"]["run_name"]
        results_dir = (
            Path("output/runs") / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"
        )

        if not results_dir.exists():
            return "pending"

        if (results_dir / "STOPPED").exists():
            return "stopped"

        if (results_dir / "FAILED").exists():
            return "failed"

        if (results_dir / "checkpoints_summary.json").exists():
            return "completed"

        if (results_dir / "RUNNING").exists():
            return "running"

        return "pending"

    def get_run_name(self, exp_id: str) -> str:
        """Get run name for experiment."""
        exp_file = Path(f"conf/experiments/{exp_id}.yaml")
        if exp_file.exists():
            exp_config = yaml.safe_load(exp_file.read_text())
            return exp_config["training_opts"]["run_name"]
        return "unknown"

    def get_output_path(self, exp_id: str) -> Path:
        """Get output path for experiment."""
        run_name = self.get_run_name(exp_id)
        return Path("output/runs") / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"

    def test_experiment_memory(self, exp_config: dict) -> Tuple[bool, str]:
        """Test if an experiment can fit in GPU memory."""
        # Add glacier_mapping to path if needed
        code_path = Path.cwd()
        if str(code_path) not in sys.path:
            sys.path.insert(0, str(code_path))

        from glacier_mapping.data.data import fetch_loaders
        from glacier_mapping.core.frame import Framework

        try:
            # Set GPU device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_rank)
            torch.cuda.set_device(0)

            # Redirect stdout to silence memory test output
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                # Initialize framework directly from config dict
                frame = Framework.from_dict(exp_config, device=0)

                # Fetch loaders
                train_loader, val_loader, test_loader = fetch_loaders(
                    processed_dir=str(exp_config["loader_opts"]["processed_dir"]),
                    batch_size=int(exp_config["loader_opts"]["batch_size"]),
                    use_channels=list(exp_config["loader_opts"]["use_channels"]),
                    output_classes=list(exp_config["loader_opts"]["output_classes"]),
                    class_names=list(exp_config["loader_opts"]["class_names"]),
                    normalize=str(exp_config["loader_opts"]["normalize"]),
                )

                # Test one training step
                for x, y_onehot, y_int in train_loader:
                    y_hat, loss = frame.optimize(x, y_onehot, y_int.squeeze(-1))
                    frame.step()
                    break

            # Clean up
            del frame
            del train_loader, val_loader, test_loader
            torch.cuda.empty_cache()
            gc.collect()

            return True, "success"

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return False, "out_of_memory"

        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            return False, f"error: {str(e)}"

    def _can_retry_error_waiting_experiment(self, exp_id: str) -> Tuple[bool, str]:
        """Direct memory test for error-waiting experiments (bypasses status checks)."""
        # Load config and test memory directly
        exp_file = Path(f"conf/experiments/{exp_id}.yaml")
        exp_config = yaml.safe_load(exp_file.read_text())

        can_run, message = self.test_experiment_memory(exp_config)

        if can_run:
            return True, "fits"
        elif message == "out_of_memory":
            return False, "oom"
        else:
            return False, message

    def can_launch_experiment(self, exp_id: str) -> Tuple[bool, str]:
        """Check if experiment can be launched."""
        exp_state = self.experiments[exp_id]

        # Skip if not pending
        if exp_state.status != "pending":
            return False, "not_pending"

        # Load config and test memory
        exp_file = Path(f"conf/experiments/{exp_id}.yaml")
        exp_config = yaml.safe_load(exp_file.read_text())

        can_run, message = self.test_experiment_memory(exp_config)

        if can_run:
            return True, "fits"
        elif message == "out_of_memory":
            return False, "oom"
        else:
            return False, message

    def launch_experiment_async(self, exp_id: str):
        """Launch experiment asynchronously."""
        exp_file = Path(f"conf/experiments/{exp_id}.yaml")
        exp_config = yaml.safe_load(exp_file.read_text())

        # Set up log file
        run_name = exp_config["training_opts"]["run_name"]
        base_output_path = Path(self.server["output_path"])
        log_dir = (
            base_output_path
            / "runs"
            / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "worker_training.log"

        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_rank)
        env["GLACIER_MAPPING_DEVICE_OVERRIDE"] = "0"

        try:
            # Open log file for writing
            log_f = open(log_file, "w")

            # Prepare config dict with output_dir injection
            exp_config_with_output = exp_config.copy()
            exp_config_with_output.setdefault("training_opts", {})["output_dir"] = str(
                log_dir
            )

            process = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "glacier_mapping.scripts.train",
                    "--config-dict",
                    json.dumps(exp_config_with_output),
                ],
                cwd=self.server["code_path"],
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Update experiment state
            exp_state = self.experiments[exp_id]
            exp_state.process = process
            exp_state.pid = process.pid
            exp_state.status = "running"
            exp_state.start_time = datetime.now()
            exp_state.log_file = log_file
            exp_state.last_log_pos = 0

            # Create running marker
            (log_dir / "RUNNING").write_text(
                f"Started: {datetime.now().isoformat()}\nPID: {process.pid}\n"
            )

            # Use display refresh flag instead of print
            self.display_needs_refresh = True
            return True

        except Exception:
            # Use display refresh flag instead of print
            self.display_needs_refresh = True
            return False

    def _detect_error_in_log(self, log_file: Path) -> Tuple[str, str, bool]:
        """
        Detect and categorize errors from log file.

        Returns:
        - error_type: "oom", "cuda", "data", "import", "config", "unknown"
        - error_message: Full error details for display
        - is_recoverable: Whether this error should be retried
        """
        if not log_file or not log_file.exists():
            return "unknown", "Log file not found", False

        error_patterns = [
            # OOM errors - recoverable
            (r"torch\.OutOfMemoryError", "oom", True, "Out of GPU memory"),
            (r"CUDA out of memory", "oom", True, "CUDA out of memory"),
            # CUDA/driver errors - recoverable
            (
                r"RuntimeError.*?CUDA.*?device-side assert",
                "cuda",
                True,
                "CUDA device assertion",
            ),
            (r"RuntimeError.*?CUDA", "cuda", True, "CUDA runtime error"),
            (r"CUDA error.*?no kernel image", "cuda", True, "CUDA kernel error"),
            # Data loading errors - recoverable
            (r"FileNotFoundError", "data", True, "File not found"),
            (r"PermissionError", "data", True, "Permission denied"),
            (r"IsADirectoryError", "data", True, "Path is directory"),
            # Import/module errors - not recoverable
            (r"ModuleNotFoundError", "import", False, "Module not found"),
            (r"ImportError", "import", False, "Import error"),
            # Configuration errors - not recoverable
            (r"KeyError", "config", False, "Configuration key error"),
            (r"TypeError.*?argument", "config", False, "Type error"),
            (r"ValueError", "config", False, "Value error"),
            # Network/IO errors - recoverable
            (r"ConnectionError", "data", True, "Connection error"),
            (r"TimeoutError", "data", True, "Timeout error"),
            (r"URLError", "data", True, "URL error"),
            # Memory allocation errors (non-OOM) - recoverable
            (r"RuntimeError.*?memory", "cuda", True, "Memory allocation error"),
            (r"alloc.*?failed", "cuda", True, "Memory allocation failed"),
        ]

        try:
            with open(log_file, "r") as f:
                log_content = f.read()

            # Look for error patterns - extract just the error line and message
            lines = log_content.strip().split("\n")
            for i, line in enumerate(lines):
                for pattern, error_type, is_recoverable, _ in error_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Get clean error message from the error line
                        error_msg = line.strip()

                        # If next line contains additional error info, include it
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if (
                                next_line
                                and not next_line.startswith(" ")
                                and not next_line.startswith("Traceback")
                            ):
                                error_msg += ": " + next_line

                        return error_type, error_msg, is_recoverable

            # Fallback: grab last few lines of log for unknown errors
            if len(lines) > 5:
                context = "\n".join(lines[-5:])
                return "unknown", context, False

            return "unknown", log_content, False

        except Exception as e:
            return "unknown", f"Failed to read log: {str(e)}", False

    def cleanup_finished_experiments(self):
        """Check and clean up finished processes."""
        for exp_id, exp_state in list(self.experiments.items()):
            if exp_state.process and exp_state.process.poll() is not None:
                # Process finished
                return_code = exp_state.process.returncode

                if exp_state.user_stopped:
                    # User stopped - set status directly, skip error detection
                    exp_state.status = "stopped"
                    self.display_needs_refresh = True
                elif return_code == 0:
                    exp_state.status = "completed"
                    self.display_needs_refresh = True
                else:
                    # Use universal error detection to categorize the failure
                    if exp_state.log_file:
                        error_type, error_message, is_recoverable = (
                            self._detect_error_in_log(exp_state.log_file)
                        )
                    else:
                        error_type, error_message, is_recoverable = (
                            "unknown",
                            "No log file available",
                            False,
                        )

                    exp_state.error_type = error_type
                    exp_state.is_recoverable = is_recoverable
                    exp_state.last_error = error_message
                    exp_state.last_error_time = datetime.now()

                    # Always start in error_waiting - let timeout logic decide when to fail
                    exp_state.status = "error_waiting"
                    exp_state.error_wait_start = datetime.now()
                    self.display_needs_refresh = True

                exp_state.stop_time = datetime.now()
                exp_state.process = None
                exp_state.pid = None

                # Remove running marker
                run_name = self.get_run_name(exp_id)
                results_marker = (
                    Path("output/runs")
                    / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"
                )
                (results_marker / "RUNNING").unlink(missing_ok=True)

                # Create appropriate marker
                if exp_state.status == "failed" and not exp_state.user_stopped:
                    (results_marker / "FAILED").write_text(
                        f"Failed: {datetime.now().isoformat()}\n{exp_state.last_error}\n"
                    )
                elif exp_state.status == "stopped":
                    (results_marker / "STOPPED").write_text(
                        f"Stopped: {datetime.now().isoformat()}\nBy user request\n"
                    )

    def handle_error_waiting_experiments(self):
        """Handle all recoverable errors - retry when conditions improve."""
        for exp_id, exp_state in self.experiments.items():
            if exp_state.status == "error_waiting":
                if exp_state.error_wait_start is None:
                    exp_state.error_wait_start = datetime.now()

                wait_time = datetime.now() - exp_state.error_wait_start

                # Retry every hour
                retry_interval = ERROR_RETRY_INTERVAL_MINUTES * 60
                if int(wait_time.total_seconds()) % retry_interval < REFRESH_INTERVAL:
                    can_launch, reason = self._can_retry_error_waiting_experiment(
                        exp_id
                    )
                    if can_launch:
                        exp_state.status = "pending"
                        exp_state.error_wait_start = None
                        exp_state.error_type = None
                        print(f"[{exp_id}] Issue resolved - ready to launch")
                        self.display_needs_refresh = True
                        continue

                if wait_time.total_seconds() > ERROR_TIMEOUT_HOURS * 3600:
                    waiting_count = len(
                        [
                            s
                            for s in self.experiments.values()
                            if s.status in ["error_waiting", "pending"]
                        ]
                    )
                    running_count = len(
                        [s for s in self.experiments.values() if s.status == "running"]
                    )

                    if waiting_count == 1 and running_count == 0:
                        exp_state.status = "failed"
                        exp_state.last_error = (
                            f"Timeout after {ERROR_TIMEOUT_HOURS}h waiting"
                        )
                        print(f"[{exp_id}] Failed: error wait timeout")
                        self.display_needs_refresh = True

    def handle_oom_experiments(self):
        """Handle OOM experiments - retry when memory frees."""
        for exp_id, exp_state in self.experiments.items():
            if exp_state.status == "oom_waiting":
                if exp_state.oom_wait_start is None:
                    exp_state.oom_wait_start = datetime.now()

                wait_time = datetime.now() - exp_state.oom_wait_start

                # Strategy 1: Periodically retry (every 30 minutes)
                if (
                    wait_time.total_seconds() % (OOM_RETRY_INTERVAL_MINUTES * 60)
                    < REFRESH_INTERVAL
                ):
                    # Try launching again (will test memory)
                    can_launch, reason = self.can_launch_experiment(exp_id)
                    if can_launch:
                        exp_state.status = "pending"  # Reset to pending
                        exp_state.oom_wait_start = None
                        print(f"[{exp_id}] Memory freed - ready to launch")
                        continue

                # Strategy 2: Hard timeout after 12 hours
                if wait_time.total_seconds() > OOM_TIMEOUT_HOURS * 3600:
                    # Check if this is the only pending/waiting experiment
                    waiting_count = len(
                        [
                            s
                            for s in self.experiments.values()
                            if s.status in ["oom_waiting", "pending"]
                        ]
                    )
                    running_count = len(
                        [s for s in self.experiments.values() if s.status == "running"]
                    )

                    # Only fail if this is the only one left AND nothing running
                    if waiting_count == 1 and running_count == 0:
                        exp_state.status = "failed"
                        exp_state.last_error = (
                            f"OOM after {OOM_TIMEOUT_HOURS}h with no memory freed"
                        )
                        exp_state.last_error_time = datetime.now()
                        print(f"[{exp_id}] Failed: OOM timeout")

    def get_experiments_by_status(self) -> Dict[str, List[ExperimentState]]:
        """Group experiments by status."""
        groups = {
            "running": [],
            "pending": [],
            "oom_waiting": [],
            "error_waiting": [],
            "stopped": [],
            "failed": [],
            "completed": [],
        }

        for exp_state in self.experiments.values():
            if exp_state.status in groups:
                groups[exp_state.status].append(exp_state)

        return groups

    def format_experiment_line(
        self, exp_state: ExperimentState, is_selected: bool
    ) -> str:
        """Format a single experiment line for display."""
        prefix = "> " if is_selected else "  "

        # Basic info
        line = f"{prefix}{exp_state.exp_id:<18}"

        if exp_state.status == "running":
            line += f" [PID: {exp_state.pid or '???':<5}]"

            # Add GPU memory if available
            if exp_state.gpu_memory_gb > 0:
                if self.total_gpu_memory_gb > 0:
                    percentage = (
                        exp_state.gpu_memory_gb / self.total_gpu_memory_gb
                    ) * 100
                    line += f" GPU: {exp_state.gpu_memory_gb:.1f}/{self.total_gpu_memory_gb:.1f}GB ({percentage:.0f}%)"
                else:
                    line += f" GPU: {exp_state.gpu_memory_gb:.1f}GB"

            # Add elapsed time
            if exp_state.start_time:
                elapsed = datetime.now() - exp_state.start_time
                hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                line += f" Time: {hours:02d}:{minutes:02d}:{seconds:02d}"

            # Add current loss if available
            if exp_state.current_loss > 0:
                line += f" Loss: {exp_state.current_loss:.3f}"

        elif exp_state.status == "pending":
            line += " Memory test: pending..."

        elif exp_state.status == "oom_waiting":
            if exp_state.oom_wait_start:
                wait_time = datetime.now() - exp_state.oom_wait_start
                hours, remainder = divmod(int(wait_time.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                line += f" Waiting: {hours:02d}:{minutes:02d}:{seconds:02d}"
            line += f" (max: {OOM_TIMEOUT_HOURS:02d}:00:00)"

        elif exp_state.status == "stopped":
            line += " Stopped by user"

        elif exp_state.status == "error_waiting":
            if exp_state.error_wait_start:
                wait_time = datetime.now() - exp_state.error_wait_start
                hours, remainder = divmod(int(wait_time.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                line += f" Waiting: {hours:02d}:{minutes:02d}:{seconds:02d}"
            line += f" (max: {ERROR_TIMEOUT_HOURS:02d}:00:00)"

            # Show error type and message (truncated for display)
            if exp_state.error_type:
                error_msg = (
                    exp_state.last_error[:80] + "..."
                    if exp_state.last_error and len(exp_state.last_error) > 80
                    else exp_state.last_error or "Unknown error"
                )
                line += f" [{exp_state.error_type.upper()}: {error_msg}]"
        elif exp_state.status == "failed":
            if exp_state.last_error:
                error_msg = (
                    exp_state.last_error[:80] + "..."
                    if len(exp_state.last_error) > 80
                    else exp_state.last_error
                )
                if exp_state.error_type:
                    line += f" [{exp_state.error_type.upper()}: {error_msg}]"
                else:
                    line += f" [Error: {error_msg}]"
            if exp_state.retry_count > 0:
                line += f" Retries: {exp_state.retry_count}/{OTHER_MAX_RETRIES}"

        elif exp_state.status == "completed":
            line += " ✓ Completed"

        return line

    def display_status(self):
        """Display main status screen."""
        self.clear_screen()

        # Header
        timestamp = datetime.now().strftime("%H:%M:%S")
        groups = self.get_experiments_by_status()

        status_line = f"Status: {'PAUSED' if self.worker_paused else 'RUNNING'}"
        status_line += (
            f" ({len(groups['running'])} active, {len(groups['pending'])} pending"
        )
        status_line += f", {len(groups['oom_waiting'])} OOM-waiting"
        status_line += f", {len(groups['error_waiting'])} error-waiting)"

        self.print_raw("=" * LINE_WIDTH)
        self.print_raw(
            f"INTERACTIVE WORKER MODE - {self.server_name} GPU {self.gpu_rank} - Last updated: {timestamp}"
        )
        self.print_raw(status_line)
        self.print_raw("=" * LINE_WIDTH)
        self.print_raw()

        # Display experiments by status
        status_order = [
            "running",
            "pending",
            "oom_waiting",
            "error_waiting",
            "stopped",
            "failed",
            "completed",
        ]
        status_labels = {
            "running": "RUNNING",
            "pending": "PENDING",
            "oom_waiting": "OOM-WAITING",
            "error_waiting": "ERROR-WAITING",
            "stopped": "STOPPED",
            "failed": "FAILED",
            "completed": "COMPLETED",
        }

        for status in status_order:
            exp_list = groups[status]
            if exp_list:
                self.print_raw(f"{status_labels[status]} ({len(exp_list)}):")
                for exp_state in exp_list:
                    is_selected = exp_state.exp_id == self.selected_exp_id
                    line = self.format_experiment_line(exp_state, is_selected)
                    self.print_raw(line)
                self.print_raw()

        # Controls
        self.print_raw("=" * LINE_WIDTH)
        self.print_raw("Experiment Controls: [s]top | [R]estart | [c]ontinue")
        self.print_raw("Worker Controls:    [p]ause | [r]efresh | [q]uit")
        self.print_raw("Navigation:         [ prev , next ]")
        self.print_raw("=" * LINE_WIDTH)

    def confirm_action(
        self,
        action_description: str,
        experiment_info: Optional[dict] = None,
        warning: Optional[str] = None,
    ) -> bool:
        """Show confirmation dialog and return True/False."""
        self.clear_screen()
        self.print_raw("=" * LINE_WIDTH)
        self.print_raw("CONFIRMATION REQUIRED")
        self.print_raw("=" * LINE_WIDTH)
        self.print_raw(f"Action: {action_description}")

        if experiment_info:
            self.print_raw(
                f"Experiment: {experiment_info['exp_id']} ({experiment_info['run_name']})"
            )
            if "output_path" in experiment_info:
                self.print_raw(f"Output path: {experiment_info['output_path']}/")

        if warning:
            self.print_raw(f"Warning: {warning}")

        self.print_raw()
        self.print_raw("Are you sure? [y]es / [n]o (default: no)")
        self.print_raw("=" * LINE_WIDTH)

        # Get user input with timeout
        start_time = time.time()
        while time.time() - start_time < CONFIRMATION_TIMEOUT:
            key = self.get_key()
            if key:
                if key.lower() == "y":
                    return True
                elif (
                    key.lower() == "n" or key == "\r" or key == "\x1b"
                ):  # Enter or Escape
                    return False
            time.sleep(0.1)

        return False  # Timeout = no

    def handle_stop_experiment(self, exp_id: str):
        """Handle stop experiment action."""
        exp_state = self.experiments[exp_id]
        exp_info = {"exp_id": exp_id, "run_name": self.get_run_name(exp_id)}

        warning = "This will terminate the running process and mark as stopped.\n"
        warning += "Note: You can resume and continue from the last checkpoint later if you want."

        if self.confirm_action("Stop running experiment", exp_info, warning):
            # Mark as user stopped BEFORE process termination
            exp_state.user_stopped = True

            if exp_state.process:
                exp_state.process.terminate()
                exp_state.status = "stopped"
                exp_state.stop_time = datetime.now()
                self.display_needs_refresh = True

                # Remove running marker and create STOPPED marker
                run_name = self.get_run_name(exp_id)
                results_marker = (
                    Path("output/runs")
                    / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"
                )
                (results_marker / "RUNNING").unlink(missing_ok=True)
                (results_marker / "STOPPED").write_text(
                    f"Stopped: {datetime.now().isoformat()}\nBy user request\n"
                )

    def handle_restart_experiment(self, exp_id: str):
        """Handle restart experiment action."""
        exp_state = self.experiments[exp_id]
        exp_info = {
            "exp_id": exp_id,
            "run_name": self.get_run_name(exp_id),
            "output_path": str(self.get_output_path(exp_id)),
        }

        warning = (
            "This will permanently delete all training progress, checkpoints, and logs!"
        )

        if self.confirm_action("Restart experiment (clean slate)", exp_info, warning):
            # Delete output directory
            output_path = self.get_output_path(exp_id)
            if output_path.exists():
                shutil.rmtree(output_path)

            # Reset state
            exp_state.status = "pending"
            exp_state.retry_count = 0
            exp_state.oom_wait_start = None
            exp_state.error_wait_start = None
            exp_state.last_error = None
            exp_state.last_error_time = None
            exp_state.error_type = None
            exp_state.user_stopped = False
            self.display_needs_refresh = True

    def handle_continue_experiment(self, exp_id: str):
        """Handle continue experiment action."""
        exp_state = self.experiments[exp_id]

        # Reset user stopped flag
        exp_state.user_stopped = False

        # Delete FAILED and STOPPED markers if they exist
        output_path = self.get_output_path(exp_id)
        (output_path / "FAILED").unlink(missing_ok=True)
        (output_path / "STOPPED").unlink(missing_ok=True)

        exp_state.status = "pending"
        self.display_needs_refresh = True

    def handle_key_input(self, key: str):
        """Handle keyboard input."""
        key_lower = key.lower()

        # WASD Navigation
        if key == "[":  # Up
            self._navigate_up()
            self.display_needs_refresh = True
        elif key == "]":  # Down
            self._navigate_down()
            self.display_needs_refresh = True

        # Experiment controls (only when experiment selected)
        elif self.selected_exp_id and key_lower == "s":
            self.handle_stop_experiment(self.selected_exp_id)
            self.display_needs_refresh = True
        elif self.selected_exp_id and key == "R":
            self.handle_restart_experiment(self.selected_exp_id)
            self.display_needs_refresh = True
        elif self.selected_exp_id and key_lower == "c":
            self.handle_continue_experiment(self.selected_exp_id)
            self.display_needs_refresh = True

        # Worker controls
        elif key_lower == "p":
            self.worker_paused = not self.worker_paused
            status = "PAUSED" if self.worker_paused else "RESUMED"
            print(f"Worker {status}")
            self.display_needs_refresh = True
        elif key_lower == "r":
            self.display_needs_refresh = True  # Just refresh display
        elif key_lower == "q":
            exp_info = {"exp_id": "worker", "run_name": "interactive worker"}
            if self.confirm_action(
                "Quit worker",
                exp_info,
                "This will stop the worker and all running experiments.",
            ):
                self.running = False
                self._cleanup_all_processes()

    def _navigate_up(self):
        """Navigate up through experiments."""
        exp_list = list(self.experiments.values())
        if not exp_list:
            return

        if self.selected_exp_id is None:
            # Select first experiment
            self.selected_exp_id = exp_list[0].exp_id
        else:
            # Find current index and select previous
            current_idx = next(
                (
                    i
                    for i, exp in enumerate(exp_list)
                    if exp.exp_id == self.selected_exp_id
                ),
                0,
            )
            if current_idx > 0:
                self.selected_exp_id = exp_list[current_idx - 1].exp_id

    def _navigate_down(self):
        """Navigate down through experiments."""
        exp_list = list(self.experiments.values())
        if not exp_list:
            return

        if self.selected_exp_id is None:
            # Select first experiment
            self.selected_exp_id = exp_list[0].exp_id
        else:
            # Find current index and select next
            current_idx = next(
                (
                    i
                    for i, exp in enumerate(exp_list)
                    if exp.exp_id == self.selected_exp_id
                ),
                0,
            )
            if current_idx < len(exp_list) - 1:
                self.selected_exp_id = exp_list[current_idx + 1].exp_id

    def _get_total_gpu_memory(self):
        """Get total GPU memory for percentage calculations."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                    f"--id={self.gpu_rank}",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.stdout.strip():
                total_mb = float(result.stdout.strip())
                self.total_gpu_memory_gb = total_mb / 1024.0
        except Exception:
            # Fallback: try without --id flag
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )

                if result.stdout.strip():
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > self.gpu_rank:
                        total_mb = float(lines[self.gpu_rank].strip())
                        self.total_gpu_memory_gb = total_mb / 1024.0
            except Exception:
                pass  # Silently fail - will show memory without percentage

    def _update_gpu_memory(self):
        """Query GPU memory for all running processes."""
        # Use simple query without GPU filtering - will get all processes
        query = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]

        try:
            result = subprocess.run(query, capture_output=True, text=True, timeout=2)

            # Parse output: "12345, 2048"
            for line in result.stdout.strip().splitlines():
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        pid = int(parts[0])
                        mem_mb = float(parts[1])
                        mem_gb = mem_mb / 1024.0

                        # Find matching experiment
                        for exp_state in self.experiments.values():
                            if exp_state.pid == pid:
                                exp_state.gpu_memory_gb = mem_gb
                                break
                    except (ValueError, IndexError):
                        continue  # Skip malformed lines

        except Exception:
            pass  # Silently fail - GPU memory monitoring is non-critical

    def _parse_training_logs(self, exp_id: str):
        """Parse recent training output for metrics."""
        exp_state = self.experiments[exp_id]

        if not exp_state.log_file or not exp_state.log_file.exists():
            return

        try:
            with open(exp_state.log_file, "r") as f:
                # Seek to last read position
                f.seek(exp_state.last_log_pos)
                new_lines = f.readlines()
                exp_state.last_log_pos = f.tell()

            # Parse patterns (most recent wins)
            for line in new_lines:
                # Pattern 1: "===== Epoch 5 Summary ====="
                epoch_match = re.search(r"Epoch (\d+) Summary", line)
                if epoch_match:
                    exp_state.current_epoch = int(epoch_match.group(1))

                # Pattern 2: "Val Ep=5 Step=10 Loss=0.234 Avg=0.245"
                loss_match = re.search(r"Avg=(\d+\.\d+)", line)
                if loss_match:
                    exp_state.current_loss = float(loss_match.group(1))

        except Exception:
            pass  # Silently fail - not critical

    def try_launch_experiments(self):
        """Try to launch pending experiments that fit in memory."""
        if self.worker_paused:
            return

        # Check if we need to wait before starting another experiment
        if self.last_experiment_start_time:
            time_since_last = datetime.now() - self.last_experiment_start_time
            if time_since_last.total_seconds() < EXPERIMENT_START_DELAY_SECONDS:
                return  # Not enough time passed yet

        # Get list of pending experiments
        pending_experiments = [
            exp_id
            for exp_id, exp_state in self.experiments.items()
            if exp_state.status == "pending"
        ]

        for exp_id in pending_experiments:
            can_launch, reason = self.can_launch_experiment(exp_id)

            if can_launch:
                if self.launch_experiment_async(exp_id):
                    self.last_experiment_start_time = datetime.now()
                    break  # Only launch one experiment per cycle
            elif reason == "oom":
                # Mark as OOM waiting
                exp_state = self.experiments[exp_id]
                if exp_state.status != "oom_waiting":
                    exp_state.status = "oom_waiting"
                    exp_state.oom_wait_start = datetime.now()
                    print(f"[{exp_id}] Out of memory, waiting...")
            else:
                # Other error - mark as error_waiting (will retry)
                exp_state = self.experiments[exp_id]
                exp_state.status = "error_waiting"
                exp_state.error_wait_start = datetime.now()
                exp_state.error_type = "config"
                exp_state.is_recoverable = True
                exp_state.last_error = f"Memory test failed: {reason}"
                exp_state.last_error_time = datetime.now()
                print(f"[{exp_id}] Memory test failed, will retry in 1h...")
                self.display_needs_refresh = True

    def run_interactive(self):
        """Main interactive worker loop."""
        # Set up terminal for non-blocking input
        import termios
        import tty

        old_settings = termios.tcgetattr(sys.stdin)
        last_refresh = time.time()

        try:
            tty.setraw(sys.stdin.fileno())

            while self.running:
                current_time = time.time()

                # Handle key input FIRST (highest priority)
                key = self.get_key()
                if key:
                    self.handle_key_input(key)
                    continue  # Skip periodic updates if key handled

                # Load experiments periodically
                if current_time - last_refresh >= REFRESH_INTERVAL:
                    self.load_experiments()
                    last_refresh = current_time
                    self.display_needs_refresh = True

                # Clean up finished processes
                self.cleanup_finished_experiments()

                # Handle all waiting errors (not just OOM)
                self.handle_error_waiting_experiments()

                # Handle OOM experiments (legacy)
                self.handle_oom_experiments()

                # Try to launch new experiments
                self.try_launch_experiments()

                # Update GPU memory for running experiments
                self._update_gpu_memory()

                # Parse logs for all running experiments
                for exp_id, exp_state in self.experiments.items():
                    if exp_state.status == "running":
                        self._parse_training_logs(exp_id)

                # Display status only when needed
                if self.display_needs_refresh:
                    self.display_status()
                    self.display_needs_refresh = False

                # Small sleep to prevent CPU spinning
                time.sleep(0.05)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.clear_screen()
            print("👋 Exiting interactive worker")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive distributed experiment worker"
    )
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
    args = parser.parse_args()

    # Git pull to get latest experiments
    git_pull()

    # Run interactive worker
    worker = InteractiveWorker(args.server, args.gpu)
    worker.run_interactive()
