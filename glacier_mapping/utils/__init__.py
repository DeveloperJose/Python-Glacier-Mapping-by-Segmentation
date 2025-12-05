"""
Utility functions for glacier mapping.

Contains utilities, and helpers.
"""

from .gpu import cleanup_gpu_memory
from .callback_utils import (
    load_dataset_metadata,
    generate_single_visualization,
    prepare_labels_for_visualization,
    extract_tiff_number,
    extract_slice_number,
    parse_slice_path,
    select_slices_by_iou_thirds,
    select_informative_test_tiles,
    log_visualizations_to_all_loggers,
)

__all__ = [
    "cleanup_gpu_memory",
    "load_dataset_metadata",
    "generate_single_visualization",
    "prepare_labels_for_visualization",
    "extract_tiff_number",
    "extract_slice_number",
    "parse_slice_path",
    "select_slices_by_iou_thirds",
    "select_informative_test_tiles",
    "log_visualizations_to_all_loggers",
]
