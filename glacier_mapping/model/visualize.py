import numpy as np
import cv2
from matplotlib import cm


COLOR_BG = np.array([128, 128, 128], dtype=np.uint8)  # Medium Gray
COLOR_CI = np.array([0, 120, 255], dtype=np.uint8)  # Saturated Blue
COLOR_DEB = np.array(
    [255, 120, 0], dtype=np.uint8
)  # Darker Orange (more orange, less yellow)
COLOR_IGNORE = np.array([0, 0, 0], dtype=np.uint8)  # Black

# Error visualization colors
COLOR_TP = np.array([0, 255, 0], dtype=np.uint8)  # Green - True Positive
COLOR_FP = np.array([255, 0, 0], dtype=np.uint8)  # Red - False Positive
COLOR_FN = np.array([255, 0, 255], dtype=np.uint8)  # Magenta - False Negative

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


def make_combined_error_mask(tp_mask, fp_mask, fn_mask, background_color=128):
    """
    Create a single RGB image with all error types combined.

    Args:
        tp_mask: Boolean array (H, W) for true positives
        fp_mask: Boolean array (H, W) for false positives
        fn_mask: Boolean array (H, W) for false negatives
        background_color: Background gray value (0-255), default 128

    Returns:
        RGB image (H, W, 3) with colored error masks:
        - TP = Green
        - FP = Red
        - FN = Magenta
        - Background = Gray
    """
    H, W = tp_mask.shape
    # Create gray background
    combined = np.full((H, W, 3), background_color, dtype=np.uint8)

    # Apply colors in order: TP, FP, FN (overlapping pixels take last applied color)
    combined[tp_mask] = COLOR_TP
    combined[fp_mask] = COLOR_FP
    combined[fn_mask] = COLOR_FN

    return combined


def make_errors_overlay(tiff_rgb, tp_mask, fp_mask, fn_mask, alpha=0.5):
    """
    Create semi-transparent overlay of error masks on TIFF RGB.

    Args:
        tiff_rgb: RGB satellite image (H, W, 3)
        tp_mask: Boolean array (H, W) for true positives
        fp_mask: Boolean array (H, W) for false positives
        fn_mask: Boolean array (H, W) for false negatives
        alpha: Transparency (0.0=invisible, 1.0=opaque)

    Returns:
        RGB image with semi-transparent error overlay
    """
    H, W = tiff_rgb.shape[:2]
    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    # Create colored overlay
    overlay[tp_mask] = COLOR_TP
    overlay[fp_mask] = COLOR_FP
    overlay[fn_mask] = COLOR_FN

    # Create alpha mask: 0 for background, alpha for error pixels
    alpha_mask = np.zeros((H, W, 1), dtype=np.float32)
    error_pixels = tp_mask | fp_mask | fn_mask
    alpha_mask[error_pixels] = alpha

    # Blend: result = tiff * (1 - alpha) + overlay * alpha
    result = (tiff_rgb * (1.0 - alpha_mask) + overlay * alpha_mask).astype(np.uint8)

    return result


def create_legend_row(labels_dict, width, height=40):
    """
    Create a horizontal legend bar with color boxes and labels.

    Args:
        labels_dict: Dictionary mapping label text to RGB color tuples
                    Example: {"Background": [128, 128, 128], "Clean Ice": [0, 120, 255]}
        width: Width of the legend bar
        height: Height of the legend bar

    Returns:
        RGB image (height, width, 3) with legend
    """
    legend = np.full((height, width, 3), 255, dtype=np.uint8)

    # Calculate box and text positions
    num_items = len(labels_dict)
    box_size = height - 16  # Leave padding
    spacing = (width - 20) // num_items  # Distribute evenly

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    x_offset = 10
    for label, color in labels_dict.items():
        # Draw color box
        box_x1 = x_offset
        box_y1 = 8
        box_x2 = box_x1 + box_size
        box_y2 = box_y1 + box_size

        cv2.rectangle(
            legend, (box_x1, box_y1), (box_x2, box_y2), tuple(int(c) for c in color), -1
        )
        cv2.rectangle(
            legend, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 1
        )  # Black border

        # Draw label text
        text_x = box_x2 + 5
        text_y = box_y1 + box_size // 2 + 4
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

        # Move to next position
        x_offset += spacing

    return legend


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


def make_redesigned_panel(
    context_rgb,
    x_rgb,
    gt_rgb,
    pr_rgb,
    gt_overlay_rgb,
    pr_overlay_rgb,
    tp_mask,
    fp_mask,
    fn_mask,
    cmap,
    class_names=None,
    metrics_text=None,
):
    """
    Create 5-row × 2-column visualization with legends.

    Layout:
        Row 1: Context (TIFF with box) | RGB Slice
        Row 2: Ground Truth | Prediction
        Row 3: GT Overlay | Pred Overlay
        Row 4: TP+FP+FN Combined | TP+FP+FN Overlay
        Row 5: False Positive | False Negative

    Args:
        context_rgb: Full TIFF with slice location box
        x_rgb: RGB satellite slice
        gt_rgb: Ground truth categorical visualization
        pr_rgb: Prediction categorical visualization
        gt_overlay_rgb: GT overlay on TIFF RGB
        pr_overlay_rgb: Prediction overlay on TIFF RGB
        tp_mask: True positive boolean mask
        fp_mask: False positive boolean mask
        fn_mask: False negative boolean mask
        cmap: Color mapping dict
        class_names: List of class names for legend
        metrics_text: Optional header text with metrics

    Returns:
        Composite visualization image with all rows and legends
    """
    # Generate error visualizations
    combined_errors = make_combined_error_mask(
        tp_mask, fp_mask, fn_mask, background_color=128
    )
    errors_overlay = make_errors_overlay(x_rgb, tp_mask, fp_mask, fn_mask, alpha=0.5)

    # Generate FP and FN individual masks
    fp_rgb = np.zeros_like(x_rgb)
    fp_rgb[fp_mask] = COLOR_FP

    fn_rgb = np.zeros_like(x_rgb)
    fn_rgb[fn_mask] = COLOR_FN

    # Build first row to get correct width with borders
    r1_temp = concat_h(
        add_title(context_rgb, "Context (Full TIFF)"),
        add_title(x_rgb, "RGB Satellite Image"),
    )

    # Get dimensions for legends
    row_width = r1_temp.shape[1]  # Full row width
    single_img_width = x_rgb.shape[1]  # Single image width (for legends on right)
    legend_height = 40

    # Create legends for each row
    # Row 2-3: Class colors (always show all classes: BG, CleanIce, Debris, Mask)
    # This ensures consistency even for binary tasks
    from collections import OrderedDict

    class_legend_dict = OrderedDict(
        [
            ("Background", COLOR_BG.tolist()),
            ("Clean Ice", COLOR_CI.tolist()),
            ("Debris", COLOR_DEB.tolist()),
            ("Mask", COLOR_IGNORE.tolist()),
        ]
    )

    class_legend = create_legend_row(class_legend_dict, single_img_width, legend_height)

    # Row 4: Error colors
    error_legend_dict = OrderedDict(
        [
            ("True Positive", COLOR_TP.tolist()),
            ("False Positive", COLOR_FP.tolist()),
            ("False Negative", COLOR_FN.tolist()),
        ]
    )
    error_legend = create_legend_row(error_legend_dict, single_img_width, legend_height)

    # Row 5: Individual error types (combined legend for both FP and FN)
    individual_error_legend = create_legend_row(
        OrderedDict(
            [
                ("False Positive", COLOR_FP.tolist()),
                ("False Negative", COLOR_FN.tolist()),
            ]
        ),
        single_img_width,
        legend_height,
    )

    # Get width for single image + border (for legends on right)
    single_img_width = r1_temp.shape[1] // 2  # Approximate single image width
    legend_height = 40

    # Create legends for each row
    # Row 2-3: Class colors (always show all classes: BG, CleanIce, Debris, Mask)
    # This ensures consistency even for binary tasks
    from collections import OrderedDict

    class_legend_dict = OrderedDict(
        [
            ("Background", COLOR_BG.tolist()),
            ("Clean Ice", COLOR_CI.tolist()),
            ("Debris", COLOR_DEB.tolist()),
            ("Mask", COLOR_IGNORE.tolist()),
        ]
    )

    class_legend = create_legend_row(class_legend_dict, single_img_width, legend_height)

    # Row 4: Error colors
    error_legend_dict = {
        "True Positive": COLOR_TP.tolist(),
        "False Positive": COLOR_FP.tolist(),
        "False Negative": COLOR_FN.tolist(),
    }
    error_legend = create_legend_row(error_legend_dict, row_width, legend_height)

    # Row 5: Individual error types (same width as combined rows)
    individual_error_legend = create_legend_row(
        {
            "False Positive": COLOR_FP.tolist(),
            "False Negative": COLOR_FN.tolist(),
        },
        row_width,
        legend_height,
    )

    # Build rows with titles and legends on the right
    # Legend needs to match height of image+title, so add title to legends too
    # Row 1: Context + RGB (no legend)
    r1 = r1_temp

    # Row 2: GT | Pred | Legend
    r2 = concat_h(
        add_title(gt_rgb, "Ground Truth"),
        add_title(pr_rgb, "Prediction"),
        add_title(class_legend, "Legend"),
    )

    # Row 3: GT Overlay | Pred Overlay | Legend
    r3 = concat_h(
        add_title(gt_overlay_rgb, "GT Overlay (semi-transparent)"),
        add_title(pr_overlay_rgb, "Pred Overlay (semi-transparent)"),
        add_title(class_legend, "Legend"),
    )

    # Row 4: Errors Overlay | Combined Errors | Legend (swapped order)
    r4 = concat_h(
        add_title(errors_overlay, "Errors Overlay on RGB"),
        add_title(combined_errors, "Combined Errors (TP+FP+FN)"),
        add_title(error_legend, "Legend"),
    )

    # Row 5: FP | FN | Legend
    r5 = concat_h(
        add_title(fp_rgb, "False Positive (predicted but not true)"),
        add_title(fn_rgb, "False Negative (true but not predicted)"),
        add_title(individual_error_legend, "Legend"),
    )

    # Combine all rows
    composite = concat_v(r1, r2, r3, r4, r5)

    # Add header with metrics
    if metrics_text is not None:
        header = title_bar(metrics_text, composite.shape[1], height=40, font_scale=0.6)
        composite = concat_v(header, composite)

    return composite
