"""Error handling utilities for glacier mapping training."""

import json
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from glacier_mapping.utils.mlflow_utils import MLflowManager
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ErrorHandler:
    """Centralized error handling and logging for glacier mapping experiments."""
    
    def __init__(self, output_dir: str, run_name: str, mlflow_logger=None):
        """
        Initialize error handler.
        
        Args:
            output_dir: Output directory for the run
            run_name: Name of the current run
            mlflow_logger: Optional MLflow logger instance
        """
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.mlflow_logger = mlflow_logger
        self.error_log_dir = self.output_dir / "error_logs"
        self.error_log_dir.mkdir(parents=True, exist_ok=True)
    
    def capture_system_info(self) -> Dict[str, Any]:
        """Capture system information at time of error."""
        try:
            # Basic system info
            system_info: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "platform": platform.platform(),
                "python_version": sys.version,
                "hostname": platform.node(),
            }
            
            # GPU info
            if torch.cuda.is_available():
                system_info["gpu_available"] = True
                system_info["gpu_count"] = torch.cuda.device_count()
                if torch.cuda.device_count() > 0:
                    system_info["gpu_name"] = torch.cuda.get_device_name(0)
                    system_info["gpu_memory_allocated"] = torch.cuda.memory_allocated(0)
                    system_info["gpu_memory_reserved"] = torch.cuda.memory_reserved(0)
            else:
                system_info["gpu_available"] = False
            
            # Memory info
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()  # type: ignore
                system_info["memory_total_gb"] = memory.total / (1024**3)
                system_info["memory_available_gb"] = memory.available / (1024**3)
                system_info["memory_percent_used"] = memory.percent
                
                # Disk info
                disk = psutil.disk_usage(str(self.output_dir))  # type: ignore
                system_info["disk_total_gb"] = disk.total / (1024**3)
                system_info["disk_free_gb"] = disk.free / (1024**3)
                system_info["disk_percent_used"] = (disk.used / disk.total) * 100
            else:
                system_info["psutil_unavailable"] = True
            
            return system_info
            
        except Exception as e:
            return {"error": f"Failed to capture system info: {str(e)}"}
    
    def log_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an exception with full context to file and MLflow.
        
        Args:
            exception: The exception that occurred
            context: Additional context information
            
        Returns:
            Path to the error log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_path = self.error_log_dir / f"error_{timestamp}.json"
        
        # Create error report
        error_report = {
            "timestamp": datetime.now().isoformat(),
            "run_name": self.run_name,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback.format_exc(),
            "system_info": self.capture_system_info(),
            "context": context or {},
        }
        
        # Save to file
        with open(error_log_path, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        # Log to MLflow if available
        if self.mlflow_logger and MLFLOW_AVAILABLE:
            try:
                # Log error as artifact
                if hasattr(self.mlflow_logger, 'experiment'):
                    self.mlflow_logger.experiment.log_artifact(
                        self.mlflow_logger.run_id,
                        str(error_log_path),
                        artifact_path="errors"
                    )
                
                # Log error metrics
                error_metrics = {
                    "error_occurred": 1,
                    "error_type_hash": hash(type(exception).__name__) % 1000000,
                }
                self.mlflow_logger.log_metrics(error_metrics)
                
            except Exception as mlflow_error:
                print(f"Warning: Failed to log error to MLflow: {mlflow_error}")
        
        # Print summary
        print(f"\n❌ ERROR LOGGED:")
        print(f"   Type: {type(exception).__name__}")
        print(f"   Message: {str(exception)}")
        print(f"   Log file: {error_log_path}")
        print(f"   Timestamp: {error_report['timestamp']}")
        
        return str(error_log_path)
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a warning with context to file and MLflow.
        
        Args:
            message: Warning message
            context: Additional context information
            
        Returns:
            Path to the warning log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        warning_log_path = self.error_log_dir / f"warning_{timestamp}.json"
        
        # Create warning report
        warning_report = {
            "timestamp": datetime.now().isoformat(),
            "run_name": self.run_name,
            "message": message,
            "system_info": self.capture_system_info(),
            "context": context or {},
        }
        
        # Save to file
        with open(warning_log_path, 'w') as f:
            json.dump(warning_report, f, indent=2)
        
        # Log to MLflow if available
        if self.mlflow_logger and MLFLOW_AVAILABLE:
            try:
                self.mlflow_logger.experiment.log_artifact(
                    self.mlflow_logger.run_id,
                    str(warning_log_path),
                    artifact_path="warnings"
                )
            except Exception as mlflow_error:
                print(f"Warning: Failed to log warning to MLflow: {mlflow_error}")
        
        # Print summary
        print(f"\n⚠️  WARNING LOGGED:")
        print(f"   Message: {message}")
        print(f"   Log file: {warning_log_path}")
        
        return str(warning_log_path)
    
    def create_error_summary(self) -> Dict[str, Any]:
        """Create a summary of all errors and warnings for this run."""
        error_files = list(self.error_log_dir.glob("error_*.json"))
        warning_files = list(self.error_log_dir.glob("warning_*.json"))
        
        summary = {
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "total_errors": len(error_files),
            "total_warnings": len(warning_files),
            "error_types": {},
            "recent_errors": [],
            "recent_warnings": [],
        }
        
        # Analyze errors
        for error_file in sorted(error_files)[-5:]:  # Last 5 errors
            try:
                with open(error_file, 'r') as f:
                    error_data = json.load(f)
                
                exc_type = error_data.get("exception_type", "Unknown")
                summary["error_types"][exc_type] = summary["error_types"].get(exc_type, 0) + 1
                
                if len(summary["recent_errors"]) < 3:
                    summary["recent_errors"].append({
                        "timestamp": error_data.get("timestamp"),
                        "type": exc_type,
                        "message": error_data.get("exception_message", "")[:100] + "..." if len(error_data.get("exception_message", "")) > 100 else error_data.get("exception_message", ""),
                    })
            except Exception:
                pass
        
        # Analyze warnings
        for warning_file in sorted(warning_files)[-3:]:  # Last 3 warnings
            try:
                with open(warning_file, 'r') as f:
                    warning_data = json.load(f)
                
                summary["recent_warnings"].append({
                    "timestamp": warning_data.get("timestamp"),
                    "message": warning_data.get("message", "")[:100] + "..." if len(warning_data.get("message", "")) > 100 else warning_data.get("message", ""),
                })
            except Exception:
                pass
        
        return summary


def setup_error_handler(output_dir: str, run_name: str, mlflow_logger=None) -> ErrorHandler:
    """
    Setup error handler for a training run.
    
    Args:
        output_dir: Output directory for the run
        run_name: Name of the current run
        mlflow_logger: Optional MLflow logger instance
        
    Returns:
        Configured ErrorHandler instance
    """
    return ErrorHandler(output_dir, run_name, mlflow_logger)