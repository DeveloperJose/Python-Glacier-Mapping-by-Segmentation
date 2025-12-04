#!/usr/bin/env python3
"""Test new visualization layout with checkpoint."""

import sys
from pathlib import Path
import tempfile
import yaml
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.lightning.glacier_datamodule import GlacierDataModule
from glacier_mapping.model.visualize import (
    build_binary_cmap,
    build_cmap_from_mask_names,
    calculate_slice_position,
    label_to_color,
    load_full_tiff_rgb,
    make_context_image,
    make_redesigned_panel,
    make_error_overlay,
    make_overlay,
    make_rgb_preview,
)
from glacier_mapping.utils.prediction import (
    calculate_binary_metrics,
    get_probabilities,
    predict_from_probs,
)


def test_visualization():
    """Generate test visualizations with new layout."""

    # Checkpoint path
    checkpoint_path = Path(
        "output/dci_velocity_desktop_20251204_110549/checkpoints/dci_velocity_desktop_20251204_110549_epoch=221_val_loss=0.0333.ckpt"
    )

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1

    print(f"Loading checkpoint: {checkpoint_path}")
    model = GlacierSegmentationModule.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    # Get processed directory from model
    processed_dir = Path(getattr(model, "processed_dir", "data/processed"))
    print(f"Processed directory: {processed_dir}")

    # Load server config for image_dir
    servers_yaml_path = Path("configs/servers.yaml")
    with open(servers_yaml_path, "r") as f:
        servers = yaml.safe_load(f)

    server_config = servers["desktop"]
    image_dir = Path(server_config.get("image_dir", ""))

    print(f"Image directory: {image_dir}")

    # Get validation data
    val_dir = processed_dir / "val"

    if not val_dir.exists():
        print(f"❌ Validation directory not found: {val_dir}")
        return 1

    # Get first 4 validation slices
    val_slices = sorted(val_dir.glob("tiff*"))[:4]

    if not val_slices:
        print("❌ No validation slices found")
        return 1

    print(f"Found {len(val_slices)} validation slices for testing")

    # Create output directory
    output_dir = Path("/tmp/viz_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get model attributes
    output_classes = getattr(model, "output_classes", [1])
    class_names = getattr(model, "class_names", ["BG", "CleanIce", "Debris"])
    metrics_opts = getattr(model, "metrics_opts", {"threshold": [0.5, 0.5]})
    threshold = metrics_opts.get("threshold", [0.5, 0.5])

    # Build colormap
    if len(output_classes) == 1:  # Binary
        cmap = build_binary_cmap(output_classes)
    else:  # Multi-class
        cmap = build_cmap_from_mask_names(class_names)

    # Load dataset metadata for context images
    import json
    import pandas as pd

    slice_meta_path = processed_dir / "slice_meta.csv"
    if slice_meta_path.exists():
        slice_meta = pd.read_csv(slice_meta_path)
        image_index_to_filename = {}
        for _, row in slice_meta.iterrows():
            img_idx = int(row["Image"])
            filename = str(row["Landsat ID"])
            if img_idx not in image_index_to_filename:
                image_index_to_filename[img_idx] = filename
    else:
        image_index_to_filename = {}

    stats_path = processed_dir / "dataset_statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
            config = stats.get("config", {})
            window_size = tuple(config.get("window_size", [512, 512]))
            overlap = config.get("overlap", 64)
    else:
        window_size = (512, 512)
        overlap = 64

    tiff_cache = {}

    # Generate visualizations
    for idx, slice_path in enumerate(val_slices, 1):
        print(f"\n[{idx}/{len(val_slices)}] Processing: {slice_path.name}")

        # Parse slice filename (tiff_NUM_slice_NUM.npy)
        parts = slice_path.name.replace(".npy", "").split("_")
        tiff_num = int(parts[1])
        slice_num = int(parts[3])

        # Load data
        x_full = np.load(slice_path)
        y_true_raw = np.load(
            slice_path.with_name(slice_path.name.replace("tiff", "mask"))
        ).astype(np.uint8)

        # Get predictions
        probs = get_probabilities(model, x_full)
        y_pred = predict_from_probs(probs, model, threshold[0] if threshold else None)

        # Prepare visualization masks
        ignore = y_true_raw == 255

        y_gt_vis = y_true_raw.copy()
        y_pred_vis = y_pred.copy()

        if len(output_classes) == 1:  # Binary
            class_idx = output_classes[0]
            y_gt_vis_binary = np.zeros_like(y_true_raw)
            y_gt_vis_binary[y_true_raw == class_idx] = 1
            y_gt_vis_binary[y_true_raw == 255] = 255
            y_gt_vis = y_gt_vis_binary

        y_pred_vis[ignore] = 255

        # Create RGB visualizations
        x_rgb = make_rgb_preview(x_full)

        # Generate context image
        context_rgb = None
        try:
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
            context_rgb = make_context_image(
                tiff_full_rgb,
                row_start,
                col_start,
                row_end,
                col_end,
                target_size=(x_rgb.shape[1], x_rgb.shape[0]),
                box_color=(255, 255, 0),
            )

        except Exception as e:
            print(f"  Warning: Could not generate context: {e}")
            from glacier_mapping.model.visualize import (
                make_error_overlay as make_error_placeholder,
            )

            context_rgb = make_error_placeholder(x_rgb.shape[:2], "Context unavailable")

        # Create overlay visualizations
        gt_overlay_rgb = make_overlay(x_rgb, y_gt_vis, cmap, alpha=0.5)
        pr_overlay_rgb = make_overlay(x_rgb, y_pred_vis, cmap, alpha=0.5)

        # TP/FP/FN masks
        tp_mask = (
            (y_pred_vis == y_gt_vis) & (~ignore) & (y_gt_vis != 0) & (y_gt_vis != 255)
        )
        fp_mask = (
            (y_pred_vis != y_gt_vis)
            & (~ignore)
            & (y_pred_vis != 0)
            & (y_pred_vis != 255)
        )
        fn_mask = (
            (y_pred_vis != y_gt_vis) & (~ignore) & (y_gt_vis != 0) & (y_gt_vis != 255)
        )

        # Calculate metrics for title
        metric_parts = []
        if len(output_classes) == 1:  # Binary
            target_class = output_classes[0]
            P, R, iou, _, _, _ = calculate_binary_metrics(
                y_pred, y_true_raw, target_class, mask=ignore
            )
            target_class_name = class_names[target_class]
            metric_parts.append(
                f"{target_class_name}: P={P:.3f} R={R:.3f} IoU={iou:.3f}"
            )

        title_text = f"TEST VIZ - TIFF {tiff_num:04d}, Slice {slice_num:02d}"
        metrics_text = title_text + " | " + " | ".join(metric_parts)

        # Create composite visualization with new layout
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
        )

        # Save visualization
        out_path = (
            output_dir / f"test_viz_{idx}_tiff{tiff_num:04d}_slice{slice_num:02d}.png"
        )
        cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        print(f"  ✅ Saved: {out_path}")

    print(f"\n✅ All visualizations saved to: {output_dir}")
    print(f"   View with: eog {output_dir}/*.png")

    return 0


if __name__ == "__main__":
    sys.exit(test_visualization())
