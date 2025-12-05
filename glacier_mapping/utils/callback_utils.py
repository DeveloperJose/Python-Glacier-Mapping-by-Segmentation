"""Callback-specific utilities that complement glacier_mapping.utils.visualize.

This module contains utilities for Lightning callbacks that handle visualization
generation, dataset metadata loading, and selection strategies. It uses functions
from glacier_mapping.utils.visualize for core visualization operations.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from glacier_mapping.utils.visualize import (
    calculate_slice_position,
    load_full_tiff_rgb,
    make_context_image,
    make_error_overlay,
    make_rgb_preview,
    make_redesigned_panel,
    build_binary_cmap,
    build_cmap_from_mask_names,
)

from glacier_mapping.utils.logging import warning
from glacier_mapping.utils.prediction import (
    calculate_binary_metrics,
    get_probabilities,
    predict_from_probs,
)

from glacier_mapping.model.metrics import IoU, precision, recall, tp_fp_fn
from glacier_mapping.utils import cleanup_gpu_memory
import torch

# Import MLflow utilities with error handling
try:
    import importlib.util

    MLFLOW_AVAILABLE = importlib.util.find_spec("mlflow") is not None
except ImportError:
    MLFLOW_AVAILABLE = False

if MLFLOW_AVAILABLE:
    from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger


def load_dataset_metadata(
    pl_module, split_type: str, image_dir: Optional[Path] = None
) -> Tuple[Dict, Tuple[int, int], int, Dict]:
    """Load dataset metadata for specific split ('val' or 'test').

    Args:
        pl_module: Lightning module with processed_dir attribute
        split_type: 'val' or 'test' split
        image_dir: Path to raw Landsat TIFF files

    Returns:
        Tuple of (image_index_to_filename, window_size, overlap, tiff_cache)
    """
    import json

    processed_dir = Path(getattr(pl_module, "processed_dir", "data/processed"))

    if not processed_dir.exists():
        warning(f"Processed directory not found: {processed_dir}")
        return {}, (512, 512), 64, {}

    # Load slice_meta.csv for specific split
    slice_meta_path = processed_dir / "slice_meta.csv"
    image_index_to_filename = {}

    if slice_meta_path.exists():
        try:
            slice_meta = pd.read_csv(slice_meta_path)
            split_meta = slice_meta[slice_meta["split"] == split_type]

            for _, row in split_meta.iterrows():
                img_idx = int(row["Image"])
                filename = str(row["Landsat ID"])
                if img_idx not in image_index_to_filename:
                    image_index_to_filename[img_idx] = filename

            # info(
            #     f"Loaded {len(image_index_to_filename)} Landsat image mappings for {split_type} split"
            # )
        except Exception as e:
            warning(f"Failed to load slice_meta.csv: {e}")
            image_index_to_filename = {}
    else:
        warning(f"slice_meta.csv not found: {slice_meta_path}")

    # Load dataset config
    stats_path = processed_dir / "dataset_statistics.json"
    window_size = (512, 512)
    overlap = 64

    if stats_path.exists():
        try:
            with open(stats_path) as f:
                stats = json.load(f)
                config = stats.get("config", {})
                window_size = tuple(config.get("window_size", [512, 512]))
                overlap = config.get("overlap", 64)
            # info(f"Loaded dataset config: window_size={window_size}, overlap={overlap}")
        except Exception as e:
            warning(f"Failed to load dataset_statistics.json: {e}")

    # Initialize TIFF cache
    tiff_cache = {}

    return image_index_to_filename, window_size, overlap, tiff_cache


def prepare_labels_for_visualization(
    y_true_raw: np.ndarray, y_pred: np.ndarray, output_classes: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare GT and pred labels for visualization (binary/multi-class handling).

    Args:
        y_true_raw: Raw ground truth labels
        y_pred: Predicted labels
        output_classes: List of target class indices

    Returns:
        Tuple of (y_gt_vis, y_pred_vis) prepared for visualization
    """
    y_gt_vis = y_true_raw.copy()
    y_pred_vis = y_pred.copy()

    if len(output_classes) == 1:  # Binary
        class_idx = output_classes[0]
        y_gt_vis_binary = np.zeros_like(y_true_raw)
        y_gt_vis_binary[y_true_raw == class_idx] = 1
        y_gt_vis_binary[y_true_raw == 255] = 255
        y_gt_vis = y_gt_vis_binary

    return y_gt_vis, y_pred_vis


def extract_tiff_number(filepath: Path) -> int:
    """Extract TIFF number from filename pattern tiff_{NUM}_slice_{SLICE}.npy

    Args:
        filepath: Path to tiff file

    Returns:
        TIFF number as integer

    Raises:
        ValueError: If TIFF number cannot be extracted
    """
    filename = filepath.name
    if not filename.startswith("tiff_"):
        raise ValueError(f"Filename does not start with 'tiff_': {filename}")

    parts = filename.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {filename}")

    try:
        tiff_num = int(parts[1])
        return tiff_num
    except (ValueError, IndexError) as e:
        raise ValueError(f"Could not extract TIFF number from {filename}: {e}")


def extract_slice_number(filepath: Path) -> int:
    """Extract slice number from filename pattern tiff_{NUM}_slice_{SLICE}.npy

    Args:
        filepath: Path to tiff file

    Returns:
        Slice number as integer

    Raises:
        ValueError: If slice number cannot be extracted
    """
    filename = filepath.name
    if not filename.startswith("tiff_"):
        raise ValueError(f"Filename does not start with 'tiff_': {filename}")

    parts = filename.replace(".npy", "").split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")

    try:
        slice_num = int(parts[3])  # Index 3 = slice number
        return slice_num
    except (ValueError, IndexError) as e:
        raise ValueError(f"Could not extract slice number from {filename}: {e}")


def parse_slice_path(filepath: Path) -> Tuple[int, int]:
    """Extract (tiff_num, slice_num) from path.

    Args:
        filepath: Path to slice file

    Returns:
        Tuple of (tiff_num, slice_num)
    """
    filename = filepath.name
    parts = filename.replace(".npy", "").split("_")
    tiff_num = int(parts[1])  # tiff_{NUM}_slice_{SLICE}
    slice_num = int(parts[3])
    return tiff_num, slice_num


def generate_single_visualization(
    x_path: Path,
    pl_module,
    output_dir: Path,
    epoch: int,
    title_prefix: str = "",
    metadata_cache: Optional[Dict] = None,
    image_dir: Optional[Path] = None,
    scale_factor: float = 0.5,
    tile_rank_map: Optional[Dict[Path, int]] = None,
    **kwargs,
) -> Path:
    """Generate single visualization using existing visualize.py functions.

    Args:
        x_path: Path to input data file
        pl_module: Lightning module
        output_dir: Directory to save visualization
        epoch: Current epoch number
        title_prefix: Prefix for title (e.g., "VAL", "TEST")
        metadata_cache: Cached metadata from load_dataset_metadata
        image_dir: Path to raw Landsat TIFF files

    Returns:
        Path to generated visualization file
    """
    # Extract metadata from cache or load fresh
    if metadata_cache:
        image_index_to_filename, window_size, overlap, tiff_cache = metadata_cache
    else:
        image_index_to_filename, window_size, overlap, tiff_cache = (
            load_dataset_metadata(
                pl_module, "val" if "val" in str(output_dir) else "test", image_dir
            )
        )

    # Extract TIFF and slice numbers
    tiff_num = extract_tiff_number(x_path)
    slice_num = extract_slice_number(x_path)

    # Load and process data
    x_full = np.load(x_path)
    y_true_raw = np.load(x_path.with_name(x_path.name.replace("tiff", "mask")))

    # Get predictions and prepare labels
    metrics_opts = getattr(pl_module, "metrics_opts", {"threshold": [0.5, 0.5]})
    threshold = metrics_opts.get("threshold", [0.5, 0.5])
    probs = get_probabilities(pl_module, x_full)
    y_pred = predict_from_probs(probs, pl_module, threshold[0] if threshold else None)

    output_classes = getattr(pl_module, "output_classes", [1])
    y_gt_vis, y_pred_vis = prepare_labels_for_visualization(
        y_true_raw, y_pred, output_classes
    )

    # Generate context image
    context_rgb = None
    tiff_filename = "unknown"

    try:
        if image_dir is None:
            raise ValueError("image_dir not provided")

        if tiff_num not in image_index_to_filename:
            raise ValueError(f"Image index {tiff_num} not in slice metadata")

        tiff_filename = image_index_to_filename[tiff_num]
        tiff_path = image_dir / tiff_filename

        if not tiff_path.exists():
            raise FileNotFoundError(f"Landsat TIFF not found: {tiff_path}")

        # Load and cache TIFF
        if tiff_num not in tiff_cache:
            tiff_cache[tiff_num] = load_full_tiff_rgb(str(tiff_path))

        tiff_full_rgb = tiff_cache[tiff_num]

        # Calculate slice position
        tiff_shape = tiff_full_rgb.shape[:2]
        row_start, col_start, row_end, col_end = calculate_slice_position(
            slice_num, tiff_shape, window_size, overlap
        )

        # Generate context with yellow box
        scaled_size = (
            int(x_full.shape[1] * scale_factor),
            int(x_full.shape[0] * scale_factor),
        )
        context_rgb = make_context_image(
            tiff_full_rgb,
            row_start,
            col_start,
            row_end,
            col_end,
            target_size=scaled_size,
            box_color=(255, 255, 0),
        )

    except (FileNotFoundError, ValueError) as e:
        warning(f"Context generation failed: {e}")
        context_rgb = make_error_overlay(
            x_full.shape[:2], f"Context unavailable:\n{str(e)[:40]}"
        )
    except Exception as e:
        warning(f"Unexpected error in context generation: {e}")
        context_rgb = make_error_overlay(x_full.shape[:2], "Context unavailable")

    # Fallback if context is still None
    if context_rgb is None:
        context_rgb = make_error_overlay(x_full.shape[:2], "Context unavailable")

    # Create RGB visualizations
    x_rgb = make_rgb_preview(x_full, scale_factor=scale_factor)

    # Calculate confidence map
    ignore = y_true_raw == 255
    if len(output_classes) == 1:  # Binary
        conf = probs[:, :, 1]  # Foreground probability
    else:  # Multi-class
        conf = np.max(probs, axis=-1)

    # Build colormap based on task type
    class_names = getattr(pl_module, "class_names", ["background", "target"])
    if len(output_classes) == 1:  # Binary
        cmap = build_binary_cmap(output_classes)
    else:  # Multi-class
        cmap = build_cmap_from_mask_names(class_names)

    # Calculate metrics for title
    metric_parts = []
    if len(output_classes) == 1:  # Binary
        target_class = output_classes[0]
        P, R, iou, _, _, _ = calculate_binary_metrics(
            y_pred, y_true_raw, target_class, mask=ignore
        )
        target_class_name = class_names[target_class]
        metric_parts.append(f"{target_class_name}: P={P:.3f} R={R:.3f} IoU={iou:.3f}")
    else:  # Multi-class
        for ci, cname in enumerate(class_names):
            if ci == 0:  # Skip background
                continue
            pred_c = (y_pred_vis == ci).astype(np.uint8)
            true_c = (y_gt_vis == ci).astype(np.uint8)
            tp_, fp_, fn_ = tp_fp_fn(torch.from_numpy(pred_c), torch.from_numpy(true_c))
            P_val = precision(tp_, fp_, fn_)
            R_val = recall(tp_, fp_, fn_)
            I_val = IoU(tp_, fp_, fn_)
            metric_parts.append(f"{cname}: P={P_val:.3f} R={R_val:.3f} IoU={I_val:.3f}")

    # Build title with rank information
    if tile_rank_map is not None and x_path in tile_rank_map:
        rank = tile_rank_map[x_path]
        # Determine rank category
        if rank <= 4:  # Top 4
            rank_text = f"#{rank} (top-{rank})"
        elif rank >= len(tile_rank_map) - 3:  # Bottom 3
            bottom_rank = rank - (len(tile_rank_map) - 3) + 1
            rank_text = f"#{rank} (bottom-{bottom_rank})"
        else:  # Middle
            rank_text = f"#{rank} (middle)"

        title_text = (
            f"{title_prefix} TIFF {tiff_num:04d}, Slice {slice_num:02d} {rank_text}"
        )
    else:
        title_text = f"{title_prefix} TIFF {tiff_num:04d}, Slice {slice_num:02d}"

    metrics_text = title_text + " | " + " | ".join(metric_parts)

    # Create confidence map visualization
    from glacier_mapping.utils.visualize import make_confidence_map

    conf_rgb = make_confidence_map(conf, invalid_mask=ignore, scale_factor=scale_factor)

    # Create composite visualization
    composite = make_redesigned_panel(
        context_rgb=context_rgb,
        x_rgb=x_rgb,
        gt_labels=y_gt_vis,
        pr_labels=y_pred_vis,
        cmap=cmap,
        class_names=class_names,
        metrics_text=metrics_text,
        conf_rgb=conf_rgb,
        mask=~ignore,  # boolean valid mask
        scale_factor=scale_factor,
    )

    # Save and return path
    tiff_dir = output_dir / f"tiff_{tiff_num:04d}"
    tiff_dir.mkdir(parents=True, exist_ok=True)
    out_path = tiff_dir / f"slice_{slice_num:02d}_epoch{epoch:04d}.png"

    import cv2

    cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    return out_path


def select_slices_by_iou_thirds(
    slice_paths: List[Path], pl_module, num_samples: int
) -> List[Path]:
    """Select validation slices using IoU-based thirds distribution.

    Args:
        slice_paths: List of available slice paths
        pl_module: Lightning module for predictions
        num_samples: Total number of samples to select

    Returns:
        List of selected slice paths
    """
    if num_samples < 3:
        return slice_paths[:num_samples]

    # Calculate IoU for each slice
    slice_ious = []
    output_classes = getattr(pl_module, "output_classes", [1])
    metrics_opts = getattr(pl_module, "metrics_opts", {"threshold": [0.5, 0.5]})
    threshold = metrics_opts.get("threshold", [0.5, 0.5])

    # info(f"Computing IoU for {len(slice_paths)} validation slices...")
    # for idx, x_path in enumerate(tqdm(slice_paths, desc="Val IoU computation")):
    for idx, x_path in enumerate(slice_paths):
        x = np.load(x_path)
        y_pred, invalid_mask = pl_module.predict_slice(x, threshold)

        y_true_raw = np.load(
            x_path.with_name(x_path.name.replace("tiff", "mask"))
        ).astype(np.uint8)

        ignore = y_true_raw == 255
        if invalid_mask is not None:
            ignore |= invalid_mask

        # Calculate IoU
        if len(output_classes) == 1:  # Binary
            target_class = output_classes[0]
            _, _, iou, _, _, _ = calculate_binary_metrics(
                y_pred, y_true_raw, target_class, mask=ignore
            )
        else:  # Multi-class
            valid = ~ignore
            y_pred_valid = y_pred[valid]
            y_true_valid = y_true_raw[valid]

            ious = []
            for ci in range(len(output_classes)):
                label = ci
                p = (y_pred_valid == label).astype(np.uint8)
                t = (y_true_valid == label).astype(np.uint8)
                tp_, fp_, fn_ = tp_fp_fn(torch.from_numpy(p), torch.from_numpy(t))
                ious.append(IoU(tp_, fp_, fn_))
            iou = np.mean(ious)

        slice_ious.append((x_path, float(iou)))

        if (idx + 1) % 20 == 0:
            cleanup_gpu_memory(synchronize=False)

    cleanup_gpu_memory()

    # Sort by IoU (descending)
    slice_ious.sort(key=lambda x: x[1], reverse=True)

    # Split into thirds
    k = num_samples // 3
    remainder = num_samples % 3

    top_k = k + (1 if remainder > 0 else 0)
    bottom_k = k + (1 if remainder > 1 else 0)
    middle_k = k

    # Select slices
    top_slices = [path for path, iou in slice_ious[:top_k]]
    bottom_slices = [path for path, iou in slice_ious[-bottom_k:]]

    middle_start = len(slice_ious) // 2 - middle_k // 2
    middle_end = middle_start + middle_k
    middle_slices = [path for path, iou in slice_ious[middle_start:middle_end]]

    selected = top_slices + middle_slices + bottom_slices

    # info(f"Selected {len(selected)} validation slices by IoU:")
    # info(
    #     f"  Top {top_k}:    {[f'{slice_ious[i][1]:.3f}' for i in range(min(top_k, len(slice_ious)))]}"
    # )
    # info(
    #     f"  Middle {middle_k}: {[f'{slice_ious[middle_start + i][1]:.3f}' for i in range(min(middle_k, len(slice_ious) - middle_start))]}"
    # )
    # info(
    #     f"  Bottom {bottom_k}: {[f'{slice_ious[len(slice_ious) - bottom_k + i][1]:.3f}' for i in range(min(bottom_k, len(slice_ious)))]}"
    # )

    return selected


def select_informative_test_tiles(
    tile_paths: List[Path], pl_module, num_samples: int
) -> Tuple[List[Path], Dict[Path, int], Dict[Path, Tuple[np.ndarray, np.ndarray]]]:
    """Select test tiles based on IoU distribution with lightweight fallback.

    Args:
        tile_paths: All available test tile paths
        pl_module: Lightning module for predictions
        num_samples: Total number of tiles to select

    Returns:
        Tuple of (selected_tiles, rank_map, prediction_cache)
    """
    # Use lightweight selection for small num_samples to avoid GPU OOM
    if num_samples > 0 and num_samples < 12:
        # info(f"Using class-pixel selection for {num_samples} tiles (lightweight mode)")
        # info("⚠️  Skipping rank computation in lightweight mode")
        selected = _select_by_class_pixels(tile_paths, pl_module, num_samples)
        return selected, {}, {}

    # Calculate IoU for each tile and cache predictions
    tile_ious = []
    prediction_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}
    output_classes = getattr(pl_module, "output_classes", [1])
    metrics_opts = getattr(pl_module, "metrics_opts", {"threshold": [0.5, 0.5]})
    threshold = metrics_opts.get("threshold", [0.5, 0.5])

    # info(f"Computing IoU for {len(tile_paths)} tiles (predictions cached for reuse)...")

    # from tqdm import tqdm
    # for idx, x_path in enumerate(tqdm(tile_paths, desc="IoU computation + caching")):
    for idx, x_path in enumerate(tile_paths):
        x = np.load(x_path)
        y_pred, invalid_mask = pl_module.predict_slice(x, threshold)

        # Cache prediction for reuse in full evaluation
        prediction_cache[x_path] = (y_pred, invalid_mask)

        y_true_raw = np.load(
            x_path.with_name(x_path.name.replace("tiff", "mask"))
        ).astype(np.uint8)

        ignore = y_true_raw == 255
        if invalid_mask is not None:
            ignore |= invalid_mask

        # Calculate IoU
        if len(output_classes) == 1:  # Binary
            target_class = output_classes[0]
            _, _, iou, _, _, _ = calculate_binary_metrics(
                y_pred, y_true_raw, target_class, mask=ignore
            )
        else:  # Multi-class - use mean IoU across classes
            valid = ~ignore
            y_pred_valid = y_pred[valid]
            y_true_valid = y_true_raw[valid]

            ious = []
            for ci in range(len(output_classes)):
                label = ci  # Compare directly: 0=BG, 1=CI, 2=Debris
                p = (y_pred_valid == label).astype(np.uint8)
                t = (y_true_valid == label).astype(np.uint8)
                tp_, fp_, fn_ = tp_fp_fn(torch.from_numpy(p), torch.from_numpy(t))
                ious.append(IoU(tp_, fp_, fn_))
            iou = np.mean(ious)

        tile_ious.append((x_path, float(iou)))

        # Periodic GPU cleanup every 20 tiles to prevent accumulation
        if (idx + 1) % 20 == 0:
            cleanup_gpu_memory(synchronize=False)

    cleanup_gpu_memory()

    # Sort by IoU (descending)
    tile_ious.sort(key=lambda x: x[1], reverse=True)

    # Split into top-K, middle-K, bottom-K
    k = num_samples // 3
    remainder = num_samples % 3

    # Distribute remainder: top gets first extra, bottom gets second
    top_k = k + (1 if remainder > 0 else 0)
    bottom_k = k + (1 if remainder > 1 else 0)
    middle_k = k

    # Select tiles
    top_tiles = [path for path, iou in tile_ious[:top_k]]
    bottom_tiles = [path for path, iou in tile_ious[-bottom_k:]]

    # Middle tiles from median region
    middle_start = len(tile_ious) // 2 - middle_k // 2
    middle_end = middle_start + middle_k
    middle_tiles = [path for path, iou in tile_ious[middle_start:middle_end]]

    selected = top_tiles + middle_tiles + bottom_tiles

    # Build rank map: {Path: absolute_rank} (1-indexed)
    rank_map = {}
    for rank, (path, iou) in enumerate(tile_ious, start=1):
        rank_map[path] = rank

    # Log IoU distribution
    # info(f"Selected {len(selected)} tiles by IoU:")
    # info(
    #     f"  Top {top_k}:    {[f'{tile_ious[i][1]:.3f}' for i in range(min(top_k, len(tile_ious)))]}"
    # )
    # info(
    #     f"  Middle {middle_k}: {[f'{tile_ious[middle_start + i][1]:.3f}' for i in range(min(middle_k, len(tile_ious) - middle_start))]}"
    # )
    # info(
    #     f"  Bottom {bottom_k}: {[f'{tile_ious[len(tile_ious) - bottom_k + i][1]:.3f}' for i in range(min(bottom_k, len(tile_ious)))]}"
    # )

    return selected, rank_map, prediction_cache


def _select_by_class_pixels(
    tile_paths: List[Path], pl_module, num_samples: int
) -> List[Path]:
    """Select tiles with most target class pixels (fallback for small num_samples)."""
    tile_class_counts = []
    output_classes = getattr(pl_module, "output_classes", [1])

    for x_path in tile_paths:
        mask_path = x_path.with_name(x_path.name.replace("tiff", "mask"))
        mask = np.load(mask_path)

        if len(output_classes) == 1:  # Binary
            class_pixels = (mask == output_classes[0]).sum()
        else:  # Multi-class
            class_pixels = ((mask > 0) & (mask != 255)).sum()

        tile_class_counts.append((x_path, int(class_pixels)))

    tile_class_counts.sort(key=lambda x: x[1], reverse=True)
    selected = [path for path, count in tile_class_counts if count > 0]

    if len(selected) < num_samples:
        selected = [path for path, count in tile_class_counts]

    return selected[:num_samples]


def log_visualizations_to_all_loggers(
    trainer,
    output_dir: Path,
    epoch: int,
    viz_type: str,  # "val_visualizations" or "test_evaluations"
):
    """Log visualizations to both MLflow and TensorBoard."""
    for logger in trainer.loggers:
        try:
            # MLflow logging
            if MLFLOW_AVAILABLE and isinstance(logger, MLFlowLogger):
                # Log CSV metrics directory if it exists
                csv_dir = output_dir / "csv_metrics"
                if csv_dir.exists():
                    logger.experiment.log_artifacts(
                        logger.run_id,
                        str(csv_dir),
                        artifact_path="csv_metrics",
                    )

                # Log TIFF directories with proper nesting
                for tile_dir in output_dir.glob("tiff_*"):
                    if tile_dir.is_dir():
                        logger.experiment.log_artifact(
                            logger.run_id,
                            str(tile_dir),
                            artifact_path=tile_dir.name,
                        )

            # TensorBoard logging
            elif isinstance(logger, TensorBoardLogger):
                import cv2
                import torch

                for tile_dir in output_dir.glob("tiff_*"):
                    if tile_dir.is_dir():
                        for png_file in tile_dir.glob("*.png"):
                            # Load PNG and convert to tensor for TensorBoard
                            img = cv2.imread(str(png_file))  # BGR format (H, W, 3)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                            img_tensor = (
                                torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                            )
                            # Use the same directory name extraction for consistency
                            dir_name = output_dir.name
                            logger.experiment.add_image(
                                f"{dir_name}/{tile_dir.name}/{png_file.stem}",
                                img_tensor,
                                global_step=epoch,
                                dataformats="CHW",
                            )

        except Exception as e:
            warning(f"Failed to log to {type(logger).__name__}: {e}")
