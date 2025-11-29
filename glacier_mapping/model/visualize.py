#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization helpers for segmentation predictions.

Uses:
 - VIRIDIS for confidence + entropy
 - Categorical colors for GT / Pred / TP / FP / FN
 - Modular 3-, 4-, 6-, 8-panel layouts
"""

import numpy as np
import cv2
from matplotlib import cm


# ============================================================
# GLOBAL COLORS (same scheme as training)
# ============================================================
COLOR_BG     = np.array([181, 101, 29], dtype=np.uint8)   # Brown
COLOR_CI     = np.array([135, 206, 250], dtype=np.uint8)  # Light Blue
COLOR_DEB    = np.array([255, 0, 0], dtype=np.uint8)      # Red
COLOR_IGNORE = np.array([0, 0, 0], dtype=np.uint8)        # Black

GLOBAL_CMAP = {
    0:   COLOR_BG,
    1:   COLOR_CI,
    2:   COLOR_DEB,
    255: COLOR_IGNORE,
}


# ============================================================
# BASIC HELPERS
# ============================================================
def make_rgb_preview(x):
    """
    Convert multispectral tile (H,W,C) into an RGB preview.

    Assumes Landsat 7 ordering:
        index 0: B1 (Blue)
        index 1: B2 (Green)
        index 2: B3 (Red)
    Other channels (NIR, SWIR, DEM, slope, physics) ignored for RGB.
    """

    # Extract R,G,B from correct indices
    R = x[..., 2]
    G = x[..., 1]
    B = x[..., 0]

    # Normalize each channel independently
    rgb_stack = np.stack([R, G, B], axis=-1).astype(np.float32)
    rgb_min = rgb_stack.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb_stack.max(axis=(0, 1), keepdims=True)
    rgb_norm = (rgb_stack - rgb_min) / (rgb_max - rgb_min + 1e-6)

    rgb_uint8 = (rgb_norm * 255).clip(0, 255).astype(np.uint8)
    return rgb_uint8

# ============================================================
# COLORMAPS
# ============================================================

def build_cmap(num_classes, is_binary, classname=None):
    """
    Build categorical colormap for GT/Pred visualization.
    
    For binary models:
        - 0 = NOT~class (brown)
        - 1 = class (blue for CleanIce, red for Debris) 
        - 255 = mask (black)
    
    For multi-class:
        - 0 = BG (brown)
        - 1 = CleanIce (blue)
        - 2 = Debris (red)
        - 255 = mask (black)
    """
    if not is_binary:
        # Full 3-class dataset
        return GLOBAL_CMAP

    # Binary case: 0 = NOT~class, 1 = class, 255 = mask
    if classname == "CleanIce":
        return {0: COLOR_BG, 1: COLOR_CI, 255: COLOR_IGNORE}
    else:  # Debris
        return {0: COLOR_BG, 1: COLOR_DEB, 255: COLOR_IGNORE}

def build_cmap_from_mask_names(mask_names):
    """
    Ensures categorical colors align with mask_names.
    
    For binary models (2 names like ["NOT~CleanIce", "CleanIce"]):
        - index 0 : NOT~class (brown)
        - index 1 : class (blue for CleanIce, red for Debris)
    
    For multi-class models (3 names like ["Background", "CleanIce", "Debris"]):
        - index 0 : background (brown)
        - index 1 : clean ice (blue)
        - index 2 : debris (red)
    
    Always adds 255 -> black for mask.
    """
    cmap = {}
    for i, name in enumerate(mask_names):
        if name.lower().startswith("bg") or name.lower().startswith("not~"):
            cmap[i] = COLOR_BG
        elif name.lower().startswith("clean"):
            cmap[i] = COLOR_CI
        elif name.lower().startswith("debr"):
            cmap[i] = COLOR_DEB
        else:
            cmap[i] = np.array([255, 255, 255], np.uint8)  # fallback white
    cmap[255] = COLOR_IGNORE
    return cmap

def label_to_color(label_img, cmap):
    """
    Convert integer label map into RGB using categorical cmap.
    """
    H, W = label_img.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, col in cmap.items():
        out[label_img == cls] = col
    return out


# ============================================================
# CONTINUOUS MAPS (VIRIDIS)
# ============================================================

def _viridis_from_scalar(scalar_01):
    """Helper: scalar [0,1] → RGB via VIRIDIS."""
    vir = cm.get_cmap("viridis")
    rgba = vir(np.clip(scalar_01, 0.0, 1.0))
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def make_confidence_map(prob, invalid_mask=None):
    """
    prob: (H,W) probability [0..1]
    Returns an RGB VIRIDIS heatmap, with invalid masked to black if given.
    """
    rgb = _viridis_from_scalar(prob)
    if invalid_mask is not None:
        rgb[invalid_mask] = 0
    return rgb


def make_entropy_map(prob_cube, invalid_mask=None):
    """
    Compute pixelwise entropy:
        H = -sum(p * log(p)), then normalized to [0,1]
    prob_cube: (H,W,C)
    """
    p = np.clip(prob_cube, 1e-8, 1.0)
    entropy = -(p * np.log(p)).sum(axis=2)

    # Normalize 0..1
    max_e = np.max(entropy) + 1e-8
    entropy_norm = entropy / max_e

    rgb = _viridis_from_scalar(entropy_norm)
    if invalid_mask is not None:
        rgb[invalid_mask] = 0
    return rgb


# ============================================================
# TP / FP / FN MASKS (BOOLEAN → RGB)
# ============================================================

def make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask):
    """
    tp_mask, fp_mask, fn_mask: boolean arrays (H,W)
    Returns:
        tp_rgb, fp_rgb, fn_rgb (uint8 RGB mask images)
    """
    H, W = tp_mask.shape
    tp_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    fp_rgb = np.zeros_like(tp_rgb)
    fn_rgb = np.zeros_like(tp_rgb)

    tp_rgb[tp_mask] = [0, 255, 0]      # green
    fp_rgb[fp_mask] = [255, 0, 0]      # red
    fn_rgb[fn_mask] = [255, 255, 0]    # yellow

    return tp_rgb, fp_rgb, fn_rgb


# ============================================================
# PANEL BUILDERS
# ============================================================

def pad_border(img, pad=4):
    """Adds a white border around image."""
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )


def concat_h(*imgs):
    imgs = [pad_border(i) for i in imgs]
    return np.concatenate(imgs, axis=1)


def concat_v(*imgs):
    return np.concatenate(imgs, axis=0)


def title_bar(text, width, height=32, font_scale=0.4):
    """
    Create a title strip above an image.
    """
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        bar, text, (10, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA
    )
    return bar


def add_title(img, text, font_scale=0.4):
    bar = title_bar(text, img.shape[1], height=32, font_scale=font_scale)
    return concat_v(bar, img)


# ============================================================
# MAJOR PANEL LAYOUTS
# ============================================================

def make_eight_panel(
    x_rgb,
    gt_rgb,
    pr_rgb,
    conf_rgb,
    tp_rgb,
    fp_rgb,
    fn_rgb,
    entropy_rgb,
    metrics_text=None,
):
    """
    2x4 dissertation layout:
        TIFF | GT | Pred | Confidence
        TP   | FP | FN   | Entropy

    metrics_text: optional string (e.g., "CI: P=0.91 R=0.87 IoU=0.80")
                  shown as a header bar across the full width.
    """
    # First row
    r1 = concat_h(
        add_title(x_rgb, "TIFF"),
        add_title(gt_rgb, "Ground Truth"),
        add_title(pr_rgb, "Prediction"),
        add_title(conf_rgb, "Confidence"),
    )

    # Second row
    r2 = concat_h(
        add_title(tp_rgb, "True Positive"),
        add_title(fp_rgb, "False Positive"),
        add_title(fn_rgb, "False Negative"),
        add_title(entropy_rgb, "Entropy"),
    )

    composite = concat_v(r1, r2)

    if metrics_text is not None:
        header = title_bar(metrics_text, composite.shape[1], height=40, font_scale=0.4)
        composite = concat_v(header, composite)

    return composite

