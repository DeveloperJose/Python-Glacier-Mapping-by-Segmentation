#!/usr/bin/env python3
"""Generate sample visualization images from checkpoint."""

import sys
import numpy as np
import cv2
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.lightning.callbacks import ValidationVisualizationCallback
from glacier_mapping.lightning.glacier_datamodule import GlacierDataModule
from glacier_mapping.utils.prediction import get_probabilities, predict_from_probs
from glacier_mapping.model.visualize import (
    make_rgb_preview,
    label_to_color,
    make_overlay,
    make_tp_fp_fn_masks,
    make_redesigned_panel,
    build_binary_cmap,
    build_cmap_from_mask_names,
    calculate_slice_position,
    load_full_tiff_rgb,
    make_context_image,
    make_error_overlay,
    make_confidence_map,
)
from glacier_mapping.model.metrics import tp_fp_fn, IoU, precision, recall
from glacier_mapping.utils.prediction import calculate_binary_metrics


def load_config(run_dir):
    """Load configuration from run directory."""
    import json

    config_path = run_dir / "conf.json"
    with open(config_path) as f:
        return json.load(f)


def find_validation_slice(val_dir, target_tiff=4, target_slice=2):
    """Find a specific validation slice."""
    target_pattern = f"tiff_{target_tiff:04d}_slice_{target_slice:02d}.npy"
    target_path = val_dir / target_pattern

    if target_path.exists():
        return target_path

    # Fallback: find any available slice
    slices = sorted(val_dir.glob("tiff_*_slice_*.npy"))
    if slices:
        print(f"Target slice not found, using: {slices[0].name}")
        return slices[0]

    raise FileNotFoundError("No validation slices found")


def generate_sample_visualization():
    """Generate sample visualization using existing checkpoint."""

    # Setup
    run_dir = Path("output/dci_velocity_desktop_20251204_110549")
    checkpoint_path = (
        run_dir
        / "checkpoints"
        / "dci_velocity_desktop_20251204_110549_epoch=222_val_loss=0.0346.ckpt"
    )
    output_dir = Path("/tmp/glacier_viz_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load configuration
    config = load_config(run_dir)
    print(f"Dataset: {config['training_opts']['dataset_name']}")
    print(f"Output classes: {config['loader_opts']['output_classes']}")
    print(f"Class names: {config['loader_opts']['class_names']}")

    # Load model
    model = GlacierSegmentationModule.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    # Setup datamodule
    loader_opts = config["loader_opts"]
    processed_dir = Path(loader_opts["processed_dir"])

    datamodule = GlacierDataModule(
        processed_dir=str(processed_dir),
        batch_size=loader_opts["batch_size"],
        landsat_channels=loader_opts["landsat_channels"],
        dem_channels=loader_opts["dem_channels"],
        spectral_indices_channels=loader_opts["spectral_indices_channels"],
        hsv_channels=loader_opts["hsv_channels"],
        physics_channels=loader_opts["physics_channels"],
        velocity_channels=loader_opts["velocity_channels"],
        output_classes=loader_opts["output_classes"],
        class_names=loader_opts["class_names"],
        normalize=loader_opts["normalize"],
    )
    datamodule.setup("fit")

    # Find validation slice
    val_dir = processed_dir / "val"
    slice_path = find_validation_slice(val_dir, target_tiff=4, target_slice=2)
    print(f"Using slice: {slice_path.name}")

    # Parse slice info
    filename = slice_path.name
    parts = filename.replace(".npy", "").split("_")
    tiff_num = int(parts[1])
    slice_num = int(parts[3])

    # Load data
    x_full = np.load(slice_path)
    y_true_raw = np.load(
        slice_path.with_name(slice_path.name.replace("tiff", "mask"))
    ).astype(np.uint8)

    print(f"Input shape: {x_full.shape}")
    print(f"Label shape: {y_true_raw.shape}")
    print(f"Unique labels: {np.unique(y_true_raw)}")

    # Get predictions
    with torch.no_grad():
        probs = get_probabilities(model, x_full)
        threshold = config["metrics_opts"]["threshold"][0]
        y_pred = predict_from_probs(probs, model, threshold)

    print(f"Prediction shape: {y_pred.shape}")
    print(f"Unique predictions: {np.unique(y_pred)}")

    # Prepare visualization masks
    ignore = y_true_raw == 255
    y_gt_vis = y_true_raw.copy()
    y_pred_vis = y_pred.copy()

    # Handle binary case
    output_classes = loader_opts["output_classes"]
    class_names = loader_opts["class_names"]

    if len(output_classes) == 1:  # Binary
        class_idx = output_classes[0]
        y_gt_vis_binary = np.zeros_like(y_true_raw)
        y_gt_vis_binary[y_true_raw == class_idx] = 1
        y_gt_vis_binary[y_true_raw == 255] = 255
        y_gt_vis = y_gt_vis_binary

    y_pred_vis[ignore] = 255

    # Build colormap
    if len(output_classes) == 1:  # Binary
        cmap = build_binary_cmap(output_classes)
    else:  # Multi-class
        cmap = build_cmap_from_mask_names(class_names)

    # Create RGB visualizations
    x_rgb = make_rgb_preview(x_full)

    # Generate context image
    context_rgb = None
    try:
        # Load dataset metadata
        import pandas as pd
        import json

        # Load slice metadata
        slice_meta_path = processed_dir / "slice_meta.csv"
        if slice_meta_path.exists():
            slice_meta = pd.read_csv(slice_meta_path)
            val_meta = slice_meta[slice_meta["split"] == "val"]
            image_index_to_filename = {}
            for _, row in val_meta.iterrows():
                img_idx = int(row["Image"])
                filename = str(row["Landsat ID"])
                if img_idx not in image_index_to_filename:
                    image_index_to_filename[img_idx] = filename

            # Get TIFF filename
            if tiff_num in image_index_to_filename:
                tiff_filename = image_index_to_filename[tiff_num]

                # Try to find TIFF in common locations
                image_dirs = [
                    Path("/home/devj/local-debian/datasets/HKH_raw/Landsat7_2005/"),
                    Path("/home/devj/local-debian/datasets/HKH/raw_landsat/"),
                    Path("/data/landsat/"),
                    Path("data/raw/"),
                ]

                tiff_path = None
                for image_dir in image_dirs:
                    potential_path = image_dir / tiff_filename
                    if potential_path.exists():
                        tiff_path = potential_path
                        break

                if tiff_path:
                    print(f"Loading TIFF: {tiff_path}")
                    tiff_full_rgb = load_full_tiff_rgb(str(tiff_path))

                    # Load dataset config for window size/overlap
                    stats_path = processed_dir / "dataset_statistics.json"
                    if stats_path.exists():
                        with open(stats_path) as f:
                            stats = json.load(f)
                            dataset_config = stats.get("config", {})
                            window_size = tuple(
                                dataset_config.get("window_size", [512, 512])
                            )
                            overlap = dataset_config.get("overlap", 64)
                    else:
                        window_size = (512, 512)
                        overlap = 64

                    # Calculate slice position
                    tiff_shape = tiff_full_rgb.shape[:2]
                    row_start, col_start, row_end, col_end = calculate_slice_position(
                        slice_num, tiff_shape, window_size, overlap
                    )

                    # Generate context
                    context_rgb = make_context_image(
                        tiff_full_rgb,
                        row_start,
                        col_start,
                        row_end,
                        col_end,
                        target_size=(x_rgb.shape[1], x_rgb.shape[0]),
                        box_color=(255, 255, 0),
                    )
                    print(
                        f"Generated context image for slice position ({row_start}, {col_start}) to ({row_end}, {col_end})"
                    )
                else:
                    print(
                        f"TIFF file not found in any standard location: {tiff_filename}"
                    )
        else:
            print("slice_meta.csv not found")

    except Exception as e:
        print(f"Context generation failed: {e}")

    # Fallback context if needed
    if context_rgb is None:
        context_rgb = make_error_overlay(
            x_rgb.shape[:2],
            f"Context unavailable\nTIFF {tiff_num:04d}, Slice {slice_num:02d}",
        )

    # Create overlays
    gt_overlay_rgb = make_overlay(x_rgb, y_gt_vis, cmap, alpha=0.5)
    pr_overlay_rgb = make_overlay(x_rgb, y_pred_vis, cmap, alpha=0.5)

    # TP/FP/FN masks
    tp_mask = (y_pred_vis == y_gt_vis) & (~ignore) & (y_gt_vis != 0) & (y_gt_vis != 255)
    fp_mask = (
        (y_pred_vis != y_gt_vis) & (~ignore) & (y_pred_vis != 0) & (y_pred_vis != 255)
    )
    fn_mask = (y_pred_vis != y_gt_vis) & (~ignore) & (y_gt_vis != 0) & (y_gt_vis != 255)

    tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

    # Calculate metrics
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

    title_text = f"VAL TIFF {tiff_num:04d}, Slice {slice_num:02d}"
    metrics_text = title_text + " | " + " | ".join(metric_parts)

    print(f"Metrics: {metrics_text}")

    # Calculate confidence map
    if len(output_classes) == 1:  # Binary
        conf = probs[:, :, 1]  # Foreground probability
    else:  # Multi-class
        conf = np.max(probs, axis=-1)

    conf_rgb = make_confidence_map(conf, invalid_mask=ignore)

    # Generate redesigned panel visualization
    print("Generating redesigned panel visualization...")
    composite = make_redesigned_panel(
        context_rgb=context_rgb,
        x_rgb=x_rgb,
        gt_rgb=label_to_color(y_gt_vis, cmap),
        pr_rgb=label_to_color(y_pred_vis, cmap),
        gt_overlay_rgb=gt_overlay_rgb,
        pr_overlay_rgb=pr_overlay_rgb,
        tp_mask=tp_mask,
        fp_mask=fp_mask,
        fn_mask=fn_mask,
        cmap=cmap,
        class_names=class_names,
        metrics_text=metrics_text,
        conf_rgb=conf_rgb,
        mask=~ignore,  # Valid pixels mask (True = valid, False = ignore)
    )

    # Save visualization
    output_path = (
        output_dir
        / f"sample_redesigned_panel_tiff{tiff_num:04d}_slice{slice_num:02d}.png"
    )
    print(f"Saving to: {output_path}")

    # Convert RGB to BGR for OpenCV
    composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(output_path), composite_bgr)

    if success:
        print(f"✅ Successfully saved sample visualization")
        print(f"📁 Output directory: {output_dir}")
        print(f"🖼️  Image size: {composite.shape}")

        # Also save individual components for reference
        components = {
            "rgb": x_rgb,
            "context": context_rgb,
            "gt": label_to_color(y_gt_vis, cmap),
            "pred": label_to_color(y_pred_vis, cmap),
            "gt_overlay": gt_overlay_rgb,
            "pred_overlay": pr_overlay_rgb,
            "tp": tp_rgb,
            "fp": fp_rgb,
            "fn": fn_rgb,
        }

        for name, img in components.items():
            comp_path = (
                output_dir
                / f"component_{name}_tiff{tiff_num:04d}_slice{slice_num:02d}.png"
            )
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(comp_path), img_bgr)

        print(f"📁 Also saved {len(components)} component images")

    else:
        print(f"❌ Failed to save visualization")

    return str(output_path) if success else None


if __name__ == "__main__":
    result = generate_sample_visualization()
    if result:
        print(f"\n🎯 Sample generated successfully!")
        print(f"📂 View images in: {Path(result).parent}")
        print(f"📸 Main file: {Path(result).name}")
    else:
        print("\n❌ Failed to generate sample")
        sys.exit(1)
