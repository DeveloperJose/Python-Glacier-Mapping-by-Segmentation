import numpy as np
import cv2
from matplotlib import cm


# -----------------------------
# Color constants
# -----------------------------

COLOR_BG = np.array([128, 128, 128], dtype=np.uint8)  # Medium Gray
COLOR_CI = np.array([0, 120, 255], dtype=np.uint8)    # Saturated Blue
COLOR_DEB = np.array([200, 80, 0], dtype=np.uint8)    # Darker Orange
COLOR_IGNORE = np.array([0, 0, 0], dtype=np.uint8)    # Black

# Error visualization colors
COLOR_TP = np.array([0, 255, 0], dtype=np.uint8)       # Green
COLOR_FP = np.array([0, 0, 255], dtype=np.uint8)       # Blue
COLOR_FN = np.array([255, 0, 255], dtype=np.uint8)     # Magenta

DEFAULT_CLASS_COLORMAP = {
    0: COLOR_BG,
    1: COLOR_CI,
    2: COLOR_DEB,
    255: COLOR_IGNORE,
}


# -----------------------------
# Utility functions
# -----------------------------

def make_rgb_preview(x):
    """Convert multispectral tile into normalized RGB preview."""
    R = x[..., 2]
    G = x[..., 1]
    B = x[..., 0]
    rgb_stack = np.stack([R, G, B], axis=-1).astype(np.float32)
    rgb_min = rgb_stack.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb_stack.max(axis=(0, 1), keepdims=True)
    rgb_norm = (rgb_stack - rgb_min) / (rgb_max - rgb_min + 1e-6)
    return (rgb_norm * 255).clip(0, 255).astype(np.uint8)


def build_cmap(num_classes, is_binary, classname=None):
    """Legacy colormap builder."""
    if not is_binary:
        return DEFAULT_CLASS_COLORMAP
    if classname == "CleanIce":
        return {0: COLOR_BG, 1: COLOR_CI, 255: COLOR_IGNORE}
    return {0: COLOR_BG, 1: COLOR_DEB, 255: COLOR_IGNORE}


def label_to_color(label_img, cmap, mask=None):
    """Convert integer label map to RGB via a categorical colormap."""
    H, W = label_img.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, col in cmap.items():
        out[label_img == cls] = col
    if mask is not None:
        out[~mask] = COLOR_IGNORE
    return out


def _viridis_from_scalar(scalar_01):
    vir = cm.get_cmap("viridis")
    rgba = vir(np.clip(scalar_01, 0.0, 1.0))
    return (rgba[..., :3] * 255).astype(np.uint8)


def make_confidence_map(prob, invalid_mask=None):
    rgb = _viridis_from_scalar(prob)
    if invalid_mask is not None:
        rgb[invalid_mask] = 0
    return rgb


def make_confidence_colorbar(width, height=40, font_scale=0.4):
    gradient = np.linspace(0, 1, width).reshape(1, -1)
    colorbar = _viridis_from_scalar(gradient)
    colorbar = np.repeat(colorbar, height - 15, axis=0)
    label_space = np.full((15, width, 3), 255, dtype=np.uint8)
    colorbar = np.vstack([colorbar, label_space])
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    cv2.putText(colorbar, "Low", (5, height - 5), font, font_scale, (0, 0, 0), thickness)
    tw = cv2.getTextSize("High", font, font_scale, thickness)[0][0]
    cv2.putText(colorbar, "High", (width - tw - 5, height - 5), font, font_scale, (0, 0, 0), thickness)
    return colorbar


def make_overlay(tiff_rgb, labels, cmap, alpha=0.5, mask=None):
    H, W = labels.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, col in cmap.items():
        if cls not in (0, 255):
            overlay[labels == cls] = col
    alpha_mask = np.zeros((H, W, 1), dtype=np.float32)
    for cls in cmap.keys():
        if cls not in (0, 255):
            alpha_mask[labels == cls] = alpha
    result = (tiff_rgb * (1 - alpha_mask) + overlay * alpha_mask).astype(np.uint8)
    if mask is not None:
        result[~mask] = COLOR_IGNORE
    return result


def pad_border(img, pad=4):
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def concat_h(*imgs):
    imgs = [pad_border(i) for i in imgs]
    return np.concatenate(imgs, axis=1)


def concat_v(*imgs):
    return np.concatenate(imgs, axis=0)


def title_bar(text, width, height=32, font_scale=0.6):
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(bar, text, (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    return bar


def add_title(img, text, font_scale=0.6):
    bar = title_bar(text, img.shape[1], height=32, font_scale=font_scale)
    return concat_v(bar, img)


def make_combined_error_mask(tp_mask, fp_mask, fn_mask, mask=None):
    H, W = tp_mask.shape
    combined = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)
    if mask is not None:
        combined[~mask] = COLOR_IGNORE
    combined[tp_mask] = COLOR_TP
    combined[fp_mask] = COLOR_FP
    combined[fn_mask] = COLOR_FN
    return combined


def make_errors_overlay(tiff_rgb, tp_mask, fp_mask, fn_mask, alpha=0.5, mask=None):
    H, W = tiff_rgb.shape[:2]
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[tp_mask] = COLOR_TP
    overlay[fp_mask] = COLOR_FP
    overlay[fn_mask] = COLOR_FN
    alpha_mask = np.zeros((H, W, 1), dtype=np.float32)
    error_pixels = tp_mask | fp_mask | fn_mask
    alpha_mask[error_pixels] = alpha
    out = (tiff_rgb * (1 - alpha_mask) + overlay * alpha_mask).astype(np.uint8)
    if mask is not None:
        out[~mask] = COLOR_IGNORE
    return out


def make_fp_fn_individual_masks(fp_mask, fn_mask, mask=None):
    H, W = fp_mask.shape
    fp_rgb = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)
    fn_rgb = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)
    if mask is not None:
        fp_rgb[~mask] = COLOR_IGNORE
        fn_rgb[~mask] = COLOR_IGNORE
    fp_rgb[fp_mask] = COLOR_FP
    fn_rgb[fn_mask] = COLOR_FN
    return fp_rgb, fn_rgb


def create_legend_row(labels_dict, width):
    num_items = len(labels_dict)
    item_height = 35
    padding = 5
    total_height = num_items * item_height + padding * 2
    legend = np.full((total_height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    box_size = 20
    y_offset = padding
    for label, color in labels_dict.items():
        box_x1 = 10
        box_y1 = y_offset + (item_height - box_size) // 2
        box_x2 = box_x1 + box_size
        box_y2 = box_y1 + box_size
        cv2.rectangle(legend, (box_x1, box_y1), (box_x2, box_y2), tuple(color.tolist()), -1)
        cv2.rectangle(legend, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 1)
        cv2.putText(
            legend,
            label,
            (box_x2 + 8, box_y1 + box_size - 5),
            font,
            font_scale,
            (0, 0, 0),
            1,
        )
        y_offset += item_height
    return legend


# ============================================================
# === make_redesigned_panel  (Option A, clean + corrected) ===
# ============================================================

def make_redesigned_panel(
    context_rgb,
    x_rgb,
    gt_labels,
    pr_labels,
    cmap,
    class_names=None,
    metrics_text=None,
    conf_rgb=None,
    mask=None,
):
    """
    Clean + correct version:
    - mask applied BEFORE error computation
    - GT/Pred recalc from masked labels
    - TP/FP/FN computed correctly
    """

    # -------------------------------
    # 1. Apply mask to labels (Option A fix)
    # -------------------------------
    gt = gt_labels.copy()
    pr = pr_labels.copy()
    if mask is not None:
        gt[~mask] = 255
        pr[~mask] = 255

    # -------------------------------
    # 2. Compute TP / FP / FN
    # -------------------------------
    tp_mask = (gt == 1) & (pr == 1)
    fp_mask = (gt != 1) & (pr == 1) & (gt != 255)
    fn_mask = (gt == 1) & (pr != 1) & (gt != 255)

    # -------------------------------
    # 3. GT/Pred RGB visualizations
    # -------------------------------
    gt_rgb = label_to_color(gt, cmap, mask=mask)
    pr_rgb = label_to_color(pr, cmap, mask=mask)

    # -------------------------------
    # 4. Overlay images
    # -------------------------------
    gt_overlay_rgb = make_overlay(x_rgb, gt, cmap, alpha=0.5, mask=mask)
    pr_overlay_rgb = make_overlay(x_rgb, pr, cmap, alpha=0.5, mask=mask)

    # -------------------------------
    # 5. Error visualizations
    # -------------------------------
    combined_errors = make_combined_error_mask(tp_mask, fp_mask, fn_mask, mask=mask)
    errors_overlay = make_errors_overlay(x_rgb, tp_mask, fp_mask, fn_mask, alpha=0.5, mask=mask)
    fp_rgb, fn_rgb = make_fp_fn_individual_masks(fp_mask, fn_mask, mask=mask)

    # -------------------------------
    # 6. Mask context_rgb and x_rgb
    # -------------------------------
    if mask is not None:
        context_masked = context_rgb.copy()
        x_masked = x_rgb.copy()
        context_masked[~mask] = COLOR_IGNORE
        x_masked[~mask] = COLOR_IGNORE
    else:
        context_masked = context_rgb
        x_masked = x_rgb

    H, W = x_rgb.shape[:2]
    empty = np.full((H, W, 3), 255, dtype=np.uint8)

    # -------------------------------
    # 7. Legends
    # -------------------------------
    from collections import OrderedDict

    class_legend_dict = OrderedDict([
        ("Background", COLOR_BG.tolist()),
        ("Clean Ice", COLOR_CI.tolist()),
        ("Debris", COLOR_DEB.tolist()),
        ("Mask", COLOR_IGNORE.tolist()),
    ])
    class_legend = add_title(create_legend_row(class_legend_dict, W), "Class Colors")

    error_legend_dict = OrderedDict([
        ("True Positive", COLOR_TP.tolist()),
        ("False Positive", COLOR_FP.tolist()),
        ("False Negative", COLOR_FN.tolist()),
    ])
    error_legend = add_title(create_legend_row(error_legend_dict, W), "Error Types")

    # -------------------------------
    # 8. Row 1
    # -------------------------------
    r1_components = [
        add_title(context_masked, "Satellite Image (RGB)"),
        add_title(x_masked, "Tested Image Slice"),
    ]
    if conf_rgb is not None:
        cbar = make_confidence_colorbar(W, height=40, font_scale=0.5)
        conf_block = concat_v(conf_rgb, cbar)
        r1_components.append(add_title(conf_block, "Confidence"))
    else:
        r1_components.append(add_title(empty, "No Confidence Data"))

    # Pad row 1 components
    maxh = max(im.shape[0] for im in r1_components)
    r1 = concat_h(*(np.pad(im, ((0, maxh - im.shape[0]), (0, 0), (0, 0)),
                        constant_values=255) if im.shape[0] != maxh else im
                    for im in r1_components))

    # -------------------------------
    # 9. Row 2
    # -------------------------------
    r2 = concat_h(
        add_title(gt_overlay_rgb, "Ground Truth Overlay"),
        add_title(pr_overlay_rgb, "Prediction Overlay"),
        class_legend,
    )

    # -------------------------------
    # 10. Row 3
    # -------------------------------
    r3 = concat_h(
        add_title(gt_rgb, "Ground Truth"),
        add_title(pr_rgb, "Prediction"),
        class_legend,
    )

    # -------------------------------
    # 11. Row 4
    # -------------------------------
    r4 = concat_h(
        add_title(errors_overlay, "Errors Overlay (TP+FP+FN)"),
        add_title(combined_errors, "Errors (TP+FP+FN)"),
        error_legend,
    )

    # -------------------------------
    # 12. Row 5
    # -------------------------------
    target_class_name = "Clean Ice" if (class_names and "Clean" in class_names[1]) else "Debris"

    r5 = concat_h(
        add_title(fp_rgb, f"False Positives (predicted {target_class_name} but incorrect)"),
        add_title(fn_rgb, f"False Negatives (was {target_class_name} but not predicted)"),
        error_legend,
    )

    # -------------------------------
    # 13. Composite
    # -------------------------------
    composite = concat_v(r1, r2, r3, r4, r5)

    if metrics_text is not None:
        header = title_bar(metrics_text, composite.shape[1], height=40, font_scale=0.6)
        composite = concat_v(header, composite)

    return composite

