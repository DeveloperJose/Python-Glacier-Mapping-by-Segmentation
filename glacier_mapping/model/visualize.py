import numpy as np
import cv2
from matplotlib import cm


# -----------------------------
# Color constants
# -----------------------------

COLOR_BG = np.array([128, 128, 128], dtype=np.uint8)  # Medium Gray
COLOR_CI = np.array([0, 120, 255], dtype=np.uint8)  # Saturated Blue
COLOR_DEB = np.array([200, 80, 0], dtype=np.uint8)  # Darker Orange
COLOR_IGNORE = np.array([0, 0, 0], dtype=np.uint8)  # Black

# Error visualization colors
COLOR_TP = np.array([0, 255, 0], dtype=np.uint8)  # Green
COLOR_FP = np.array([0, 0, 255], dtype=np.uint8)  # Blue
COLOR_FN = np.array([255, 0, 255], dtype=np.uint8)  # Magenta

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


def build_binary_cmap(output_classes, class_names=None):
    """
    Build colormap for binary models based on target class.

    For binary models, predictions are 0/1, where:
    - 0 = background (not target class)
    - 1 = target class (CleanIce or Debris)

    This function maps index 1 to the correct color based on which class
    is being segmented (output_classes[0]).

    Args:
        output_classes: List with single target class index (e.g., [1] or [2])
        class_names: Optional class names (unused, for API compatibility)

    Returns:
        Colormap dict: {0: BG_color, 1: target_color, 255: ignore_color}
    """
    target_class = output_classes[0]

    if target_class == 1:  # Clean Ice binary task
        return {0: COLOR_BG, 1: COLOR_CI, 255: COLOR_IGNORE}
    elif target_class == 2:  # Debris binary task
        return {0: COLOR_BG, 1: COLOR_DEB, 255: COLOR_IGNORE}
    else:
        # Fallback for any other binary task
        return {0: COLOR_BG, 1: COLOR_CI, 255: COLOR_IGNORE}


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
    """
    Create a horizontal colorbar for confidence visualization.

    Args:
        width: Width of the colorbar
        height: Height of the colorbar (default 40)
        font_scale: Font scale for labels (default 0.4)

    Returns:
        RGB image (height, width, 3) with colorbar and labels
    """
    # Create gradient from 0 to 1
    gradient = np.linspace(0, 1, width).reshape(1, -1)
    colorbar = _viridis_from_scalar(gradient)

    # Expand to full height
    colorbar = np.repeat(colorbar, height - 15, axis=0)  # Leave space for labels

    # Add white space for labels at bottom
    label_space = np.full((15, width, 3), 255, dtype=np.uint8)
    colorbar = np.vstack([colorbar, label_space])

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    # Low confidence label
    cv2.putText(
        colorbar,
        "Low",
        (5, height - 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    # High confidence label
    text_width = cv2.getTextSize("High", font, font_scale, thickness)[0][0]
    cv2.putText(
        colorbar,
        "High",
        (width - text_width - 5, height - 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

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
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )


def concat_h(*imgs):
    # First pad borders
    imgs = [pad_border(i) for i in imgs]

    # Then match heights by padding with white pixels
    maxh = max(im.shape[0] for im in imgs)
    imgs = [
        np.pad(im, ((0, maxh - im.shape[0]), (0, 0), (0, 0)), constant_values=255)
        if im.shape[0] != maxh
        else im
        for im in imgs
    ]

    return np.concatenate(imgs, axis=1)


def concat_v(*imgs):
    return np.concatenate(imgs, axis=0)


def title_bar(text, width, height=32, font_scale=0.6):
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        bar, text, (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2
    )
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
    """
    Create individual FP and FN masks with gray background and black masked areas.

    Args:
        fp_mask: Boolean array (H, W) for false positives
        fn_mask: Boolean array (H, W) for false negatives
        mask: Optional boolean mask for valid pixels (True = valid, False = ignore)

    Returns:
        Tuple of (fp_rgb, fn_rgb) with gray background and colored error pixels
    """
    H, W = fp_mask.shape

    # Create gray background
    fp_rgb = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)
    fn_rgb = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)

    # Apply mask if provided (set masked areas to black)
    if mask is not None:
        fp_rgb[~mask] = COLOR_IGNORE
        fn_rgb[~mask] = COLOR_IGNORE

    # Apply colored error pixels
    fp_rgb[fp_mask] = COLOR_FP
    fn_rgb[fn_mask] = COLOR_FN

    return fp_rgb, fn_rgb


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

    tp_rgb[tp_mask] = [0, 255, 0]  # Green
    fp_rgb[fp_mask] = [255, 0, 0]  # Red
    fn_rgb[fn_mask] = [255, 0, 255]  # Magenta

    return tp_rgb, fp_rgb, fn_rgb


def create_legend_row(labels_dict, width, height=40):
    """
    Create a vertical legend with color boxes and labels stacked.

    Args:
        labels_dict: Dictionary mapping label text to RGB color tuples
                    Example: {"Background": [128, 128, 128], "Clean Ice": [0, 120, 255]}
        width: Width of the legend (will be used fully)
        height: Height per item (default 40, but will be overridden to fit all items)

    Returns:
        RGB image (total_height, width, 3) with vertical legend
    """
    num_items = len(labels_dict)
    item_height = 35  # Height per legend item
    padding = 5
    total_height = num_items * item_height + padding * 2

    legend = np.full((total_height, width, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1

    box_size = 20  # Color box size
    y_offset = padding

    for label, color in labels_dict.items():
        # Draw color box (left side)
        box_x1 = 10
        box_y1 = y_offset + (item_height - box_size) // 2
        box_x2 = box_x1 + box_size
        box_y2 = box_y1 + box_size

        cv2.rectangle(
            legend, (box_x1, box_y1), (box_x2, box_y2), tuple(int(c) for c in color), -1
        )
        cv2.rectangle(
            legend, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 1
        )  # Black border

        # Draw label text (right of box)
        text_x = box_x2 + 8
        text_y = y_offset + item_height // 2 + 5
        cv2.putText(
            legend,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

        # Move to next row
        y_offset += item_height

    return legend


def calculate_slice_position(slice_num, tiff_shape, window_size, overlap):
    """
    Calculate slice position in full TIFF from slice number.

    The slice numbering matches the slicing loop in slice.py:
        for row in range(0, height, stride):
            for col in range(0, width, stride):
                # slice_num increments here

    This means edge slices (partial windows) are included in the count.

    Args:
        slice_num: Sequential slice number (0, 1, 2, ...)
        tiff_shape: (height, width) of full TIFF
        window_size: (h, w) slice window size
        overlap: Overlap between slices

    Returns:
        (row_start, col_start, row_end, col_end)
    """
    stride_h = window_size[0] - overlap
    stride_w = window_size[1] - overlap

    # Calculate number of slices per row - must match slice.py loop:
    # len(range(0, width, stride)) = ceil(width / stride)
    # This includes partial edge slices, unlike the old formula which only
    # counted full-size slices: (width - window) // stride + 1
    num_cols = (tiff_shape[1] + stride_w - 1) // stride_w

    # Calculate row and column from slice number
    slice_row = slice_num // num_cols
    slice_col = slice_num % num_cols

    # Calculate pixel positions
    row_start = slice_row * stride_h
    col_start = slice_col * stride_w
    row_end = min(row_start + window_size[0], tiff_shape[0])
    col_end = min(col_start + window_size[1], tiff_shape[1])

    return (row_start, col_start, row_end, col_end)


def load_full_tiff_rgb(tiff_path):
    """
    Load full TIFF and convert to RGB preview.

    Args:
        tiff_path: Path to original TIFF file

    Returns:
        RGB array (H, W, 3) uint8
    """
    import rasterio

    with rasterio.open(tiff_path) as src:
        # Read RGB bands (Landsat ordering: B, G, R in first 3 channels)
        # Read bands 3, 2, 1 (Red, Green, Blue)
        rgb = src.read([3, 2, 1])  # Read as R, G, B

        # Transpose to (H, W, 3)
        rgb = np.transpose(rgb, (1, 2, 0))

        # Normalize to uint8
        rgb = rgb.astype(np.float32)
        rgb_min = rgb.min()
        rgb_max = rgb.max()
        rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-6)
        rgb_uint8 = (rgb_norm * 255).clip(0, 255).astype(np.uint8)

        return rgb_uint8


def make_context_image(
    tiff_full_rgb,
    slice_row_start,
    slice_col_start,
    slice_row_end,
    slice_col_end,
    target_size=(512, 512),
    box_color=(255, 255, 0),
):
    """
    Create context image showing full TIFF with slice location box.

    Args:
        tiff_full_rgb: Full TIFF as RGB array (H, W, 3)
        slice_row_start: Slice starting row
        slice_col_start: Slice starting column
        slice_row_end: Slice ending row
        slice_col_end: Slice ending column
        target_size: Output image size (downsampled)
        box_color: Color for slice location box (RGB), default yellow

    Returns:
        RGB image with slice location box (uint8)
    """
    H, W = tiff_full_rgb.shape[:2]

    # Downsample full TIFF to target size
    context = cv2.resize(tiff_full_rgb, target_size, interpolation=cv2.INTER_AREA)

    # Calculate box position in downsampled image
    scale_h = target_size[0] / H
    scale_w = target_size[1] / W

    box_r1 = int(slice_row_start * scale_h)
    box_c1 = int(slice_col_start * scale_w)
    box_r2 = int(slice_row_end * scale_h)
    box_c2 = int(slice_col_end * scale_w)

    # Draw rectangle (thick border for visibility)
    thickness = max(2, int(target_size[0] / 128))  # Adaptive thickness
    cv2.rectangle(context, (box_c1, box_r1), (box_c2, box_r2), box_color, thickness)

    return context


def make_error_overlay(shape, error_message):
    """
    Create gray placeholder with white error message for missing context.

    Args:
        shape: (height, width) for image
        error_message: Text to display on image (use \n for multiple lines)

    Returns:
        RGB array (H, W, 3) uint8 with error text
    """
    import cv2

    # Create gray background
    img = np.full((*shape, 3), 128, dtype=np.uint8)

    # Add error text (centered, white)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)  # White text

    # Split message into lines
    lines = error_message.split("\n")
    line_height = 25
    y_start = shape[0] // 2 - (len(lines) * line_height) // 2

    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x = (shape[1] - text_size[0]) // 2
        y = y_start + i * line_height
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return img


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
    errors_overlay = make_errors_overlay(
        x_rgb, tp_mask, fp_mask, fn_mask, alpha=0.5, mask=mask
    )
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

    class_legend_dict = OrderedDict(
        [
            ("Background", COLOR_BG.tolist()),
            ("Clean Ice", COLOR_CI.tolist()),
            ("Debris", COLOR_DEB.tolist()),
            ("Mask", COLOR_IGNORE.tolist()),
        ]
    )
    class_legend = add_title(create_legend_row(class_legend_dict, W), "Class Colors")

    error_legend_dict = OrderedDict(
        [
            ("True Positive", COLOR_TP.tolist()),
            ("False Positive", COLOR_FP.tolist()),
            ("False Negative", COLOR_FN.tolist()),
        ]
    )
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
    r1 = concat_h(
        *(
            np.pad(im, ((0, maxh - im.shape[0]), (0, 0), (0, 0)), constant_values=255)
            if im.shape[0] != maxh
            else im
            for im in r1_components
        )
    )

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
    target_class_name = (
        "Clean Ice" if (class_names and "Clean" in class_names[1]) else "Debris"
    )

    r5 = concat_h(
        add_title(
            fp_rgb, f"False Positives (predicted {target_class_name} but incorrect)"
        ),
        add_title(
            fn_rgb, f"False Negatives (was {target_class_name} but not predicted)"
        ),
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
