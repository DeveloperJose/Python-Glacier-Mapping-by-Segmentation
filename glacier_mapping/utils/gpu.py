"""GPU memory management utilities for glacier mapping."""

import gc

import torch


def cleanup_gpu_memory(synchronize: bool = True) -> None:
    """Clean up GPU memory to prevent OOM errors.

    This function performs aggressive GPU memory cleanup by:
    1. Emptying PyTorch's GPU cache
    2. Optionally synchronizing CUDA operations
    3. Running Python garbage collection

    Args:
        synchronize: If True, synchronize CUDA operations before cleanup.
                    Set to False for lightweight cleanup during loops.
                    Default is True for maximum cleanup between operations.

    Usage:
        # Standard cleanup (with synchronization)
        cleanup_gpu_memory()

        # Lightweight cleanup in loops
        cleanup_gpu_memory(synchronize=False)

        # Periodic cleanup every N iterations
        if (idx + 1) % 20 == 0:
            cleanup_gpu_memory(synchronize=False)
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if synchronize:
            torch.cuda.synchronize()
    gc.collect()
