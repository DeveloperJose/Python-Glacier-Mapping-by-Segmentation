import numpy as np
import cv2
from matplotlib import cm


COLOR_BG = np.array([128, 128, 128], dtype=np.uint8)  # Medium Gray
COLOR_CI = np.array([0, 120, 255], dtype=np.uint8)  # Saturated Blue
COLOR_DEB = np.array([255, 165, 0], dtype=np.uint8)  # Pure Orange
COLOR_IGNORE = np.array([0, 0, 0], dtype=np.uint8)  # Black

DEFAULT_CLASS_COLORMAP = {
    0: COLOR_BG,
    1: COLOR_CI,
    2: COLOR_DEB,
    255: COLOR_IGNORE,
}


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
        return DEFAULT_CLASS_COLORMAP

    # Binary case: 0 = NOT~class, 1 = class, 255 = mask
    if classname == "CleanIce":
        return {0: COLOR_BG, 1: COLOR_CI, 255: COLOR_IGNORE}
    else:  # Debris
        return {0: COLOR_BG, 1: COLOR_DEB, 255: COLOR_IGNORE}


def build_binary_cmap(output_classes, class_names=None):
    """
    Build colormap for binary models based on target class.

    For binary models, predictions are 0/1, where:
    - 0 = background (not the target class)
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


def label_to_color(label_img, cmap):
    """
    Convert integer label map into RGB using categorical cmap.
    """
    H, W = label_img.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, col in cmap.items():
        out[label_img == cls] = col
    return out


def _viridis_from_scalar(scalar_01):
    """Helper: scalar [0,1] → RGB via VIRIDIS."""
    vir = cm.get_cmap("viridis")
    rgba = vir(np.clip(scalar_01, 0.0, 1.0))
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def make_confidence_map(prob, invalid_mask=None):
    """Convert probability map to RGB VIRIDIS heatmap."""
    rgb = _viridis_from_scalar(prob)
    if invalid_mask is not None:
        rgb[invalid_mask, :] = 0
    return rgb


def make_entropy_map(prob_cube, invalid_mask=None):
    """Compute pixelwise entropy H = -sum(p * log(p)) and convert to RGB VIRIDIS heatmap."""
    p = np.clip(prob_cube, 1e-8, 1.0)

    # Handle both 2D (single class) and 3D (multi-class) probability cubes
    if len(p.shape) == 2:
        entropy = -(p * np.log(p))
    else:
        entropy = -(p * np.log(p)).sum(axis=-1)

    max_e = np.max(entropy) + 1e-8
    entropy_norm = entropy / max_e

    rgb = _viridis_from_scalar(entropy_norm)
    if invalid_mask is not None:
        rgb[invalid_mask, :] = 0
    return rgb


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
        shape: (height, width) for the image
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


def make_overlay(tiff_rgb, label_img, cmap, alpha=0.5):
    """
    Create semi-transparent overlay of labels on TIFF image.

    Background (class 0) is completely invisible, showing pure TIFF.
    Glacier classes are rendered as semi-transparent colored overlays.

    Args:
        tiff_rgb: RGB satellite image (H, W, 3)
        label_img: Label mask (H, W) with class indices
        cmap: Color mapping {class_idx: [R, G, B]}
        alpha: Transparency (0.0=invisible, 1.0=opaque). Default 0.5.

    Returns:
        RGB image with semi-transparent overlay (uint8)
    """
    H, W = label_img.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    # Create colored overlay (BG class will be invisible)
    for cls, col in cmap.items():
        if cls == 0:  # Skip background - leave transparent
            continue
        if cls == 255:  # Skip ignore mask
            continue
        overlay[label_img == cls] = col

    # Create alpha mask: 0 for BG/ignore, alpha for classes
    alpha_mask = np.zeros((H, W, 1), dtype=np.float32)
    for cls in cmap.keys():
        if cls != 0 and cls != 255:  # Only non-BG, non-ignore classes
            alpha_mask[label_img == cls] = alpha

    # Blend: result = tiff * (1 - alpha) + overlay * alpha
    result = (tiff_rgb * (1.0 - alpha_mask) + overlay * alpha_mask).astype(np.uint8)

    return result


def pad_border(img, pad=4):
    """Adds a white border around image."""
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )


def concat_h(*imgs):
    imgs = [pad_border(i) for i in imgs]
    return np.concatenate(imgs, axis=1)


def concat_v(*imgs):
    return np.concatenate(imgs, axis=0)


def title_bar(text, width, height=32, font_scale=0.6):
    """Create a title strip with text."""
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (10, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return bar


def add_title(img, text, font_scale=0.6):
    bar = title_bar(text, img.shape[1], height=32, font_scale=font_scale)
    return concat_v(bar, img)


def make_eight_panel(
    x_rgb,
    gt_rgb,
    pr_rgb,
    gt_overlay_rgb,
    pr_overlay_rgb,
    tp_rgb,
    fp_rgb,
    fn_rgb,
    conf_rgb=None,  # Kept for backward compatibility, not displayed
    entropy_rgb=None,  # Kept for backward compatibility, not displayed
    metrics_text=None,
):
    """
    Create 2x4 panel visualization:
        Row 1: TIFF | GT | Pred | GT Overlay
        Row 2: TP   | FP | FN   | Pred Overlay

    Args:
        x_rgb: RGB satellite image
        gt_rgb: Ground truth categorical visualization
        pr_rgb: Prediction categorical visualization
        gt_overlay_rgb: TIFF with semi-transparent GT overlay
        pr_overlay_rgb: TIFF with semi-transparent prediction overlay
        tp_rgb: True positive mask
        fp_rgb: False positive mask
        fn_rgb: False negative mask
        conf_rgb: (deprecated, not displayed)
        entropy_rgb: (deprecated, not displayed)
        metrics_text: Optional header text with metrics
    """
    r1 = concat_h(
        add_title(x_rgb, "TIFF"),
        add_title(gt_rgb, "Ground Truth"),
        add_title(pr_rgb, "Prediction"),
        add_title(gt_overlay_rgb, "Context"),
    )

    r2 = concat_h(
        add_title(tp_rgb, "True Positive"),
        add_title(fp_rgb, "False Positive"),
        add_title(fn_rgb, "False Negative"),
        add_title(pr_overlay_rgb, "Pred Overlay"),
    )

    composite = concat_v(r1, r2)

    if metrics_text is not None:
        header = title_bar(metrics_text, composite.shape[1], height=40, font_scale=0.6)
        composite = concat_v(header, composite)

    return composite
