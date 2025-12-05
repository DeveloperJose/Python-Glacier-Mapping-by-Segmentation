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


def scale_image(img, scale_factor=0.5):
    """Scale image by factor using cv2.resize.

    Args:
        img: Input image array (H, W, C) or (H, W)
        scale_factor: Scaling factor (0.5 = half size, 1.0 = original)

    Returns:
        Scaled image array
    """
    if scale_factor == 1.0:
        return img

    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))

    if len(img.shape) == 3:
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    elif len(img.shape) == 2:
        return cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")


def make_rgb_preview(x, scale_factor=0.5):
    """Convert multispectral tile into normalized RGB preview."""
    import glacier_mapping.utils.logging as log

    # Strict validation: require exactly 3 channels for RGB preview
    if x.shape[-1] < 3:
        error_msg = f"make_rgb_preview requires at least 3 channels for RGB preview, got {x.shape[-1]} channels. Input shape: {x.shape}"
        log.error(error_msg)
        raise ValueError(error_msg)

    log.debug(f"make_rgb_preview: processing input with shape {x.shape}")
    R = x[..., 2]
    G = x[..., 1]
    B = x[..., 0]
    rgb_stack = np.stack([R, G, B], axis=-1).astype(np.float32)
    rgb_min = rgb_stack.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb_stack.max(axis=(0, 1), keepdims=True)
    rgb_norm = (rgb_stack - rgb_min) / (rgb_max - rgb_min + 1e-6)
    rgb_uint8 = (rgb_norm * 255).clip(0, 255).astype(np.uint8)
    return scale_image(rgb_uint8, scale_factor)


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
    print(f"DEBUG CMAP: mask_names={mask_names}, len={len(mask_names)}")
    cmap = {}
    for i, name in enumerate(mask_names):
        print(f"DEBUG CMAP: i={i}, name={name}")
        name_lower = name.lower()

        if name_lower.startswith("bg") or name_lower.startswith("not~"):
            cmap[i] = COLOR_BG
        elif name_lower.startswith("clean"):
            cmap[i] = COLOR_CI
        elif name_lower.startswith("debr"):
            cmap[i] = COLOR_DEB
        else:
            # For multi-class, use default mapping based on index
            if len(mask_names) == 3:  # Multi-class scenario
                if i == 0:
                    cmap[i] = COLOR_BG
                elif i == 1:
                    cmap[i] = COLOR_CI
                elif i == 2:
                    cmap[i] = COLOR_DEB
                else:
                    cmap[i] = np.array([255, 255, 255], np.uint8)
            else:
                cmap[i] = np.array([255, 255, 255], np.uint8)

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


def make_confidence_map(prob, invalid_mask=None, scale_factor=0.5):
    rgb = _viridis_from_scalar(prob)
    if invalid_mask is not None:
        rgb[invalid_mask] = 0
    return scale_image(rgb, scale_factor)


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

    # Add text labels with better rendering
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(1 * font_scale / 0.4))
    text_color = (40, 40, 40)  # Softer black

    # Low confidence label
    cv2.putText(
        colorbar,
        "Low",
        (8, height - 6),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

    # High confidence label with better positioning
    (text_w, text_h), baseline = cv2.getTextSize("High", font, font_scale, thickness)
    cv2.putText(
        colorbar,
        "High",
        (width - text_w - 8, height - 6),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

    return colorbar


def make_overlay(tiff_rgb, labels, cmap, alpha=0.5, mask=None, scale_factor=0.5):
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
    return scale_image(result, scale_factor)


def pad_border(img, pad=4, value=(255, 255, 255)):
    """Add a border to an image with a specified color."""
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)


def title_bar(text, width, height=32, font_scale=0.6):
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    # Use better font rendering with anti-aliasing and improved positioning
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * font_scale / 0.6))  # Scale thickness with font
    color = (20, 20, 20)  # Softer black instead of pure black

    # Get text size for better centering
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Position text with better vertical centering
    x = 10
    y = height // 2 + text_h // 2

    cv2.putText(bar, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return bar


def add_title(img, text, font_scale=0.6):
    bar = title_bar(text, img.shape[1], height=32, font_scale=font_scale)
    return _stack_images([bar, img], axis=0)


def make_combined_error_mask(tp_mask, fp_mask, fn_mask, mask=None):
    H, W = tp_mask.shape
    combined = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)
    if mask is not None:
        combined[~mask] = COLOR_IGNORE
    combined[tp_mask] = COLOR_TP
    combined[fp_mask] = COLOR_FP
    combined[fn_mask] = COLOR_FN
    return combined


def make_errors_overlay(
    tiff_rgb, tp_mask, fp_mask, fn_mask, alpha=0.5, mask=None, scale_factor=0.5
):
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
    return scale_image(out, scale_factor)


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


def create_legend_row(labels_dict, width, height=40, scale_factor=0.5):
    """
    Create a vertical legend with color boxes and labels stacked.

    Args:
        labels_dict: Dictionary mapping label text to RGB color tuples
                    Example: {"Background": [128, 128, 128], "Clean Ice": [0, 120, 255]}
        width: Width of the legend (will be used fully)
        height: Height per item (default 40, but will be overridden to fit all items)
        scale_factor: Scale factor for font sizing

    Returns:
        RGB image (total_height, width, 3) with vertical legend
    """
    num_items = len(labels_dict)
    item_height = max(30, int(30 * scale_factor))  # Scale item height
    padding = max(5, int(5 * scale_factor))
    total_height = num_items * item_height + padding * 2

    legend = np.full((total_height, width, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, 0.45 * scale_factor)  # Scale font size
    thickness = max(1, int(1 * scale_factor))

    box_size = max(15, int(18 * scale_factor))  # Scale box size
    y_offset = padding

    for label, color in labels_dict.items():
        # Draw color box (left side)
        box_x1 = padding
        box_y1 = y_offset + (item_height - box_size) // 2
        box_x2 = box_x1 + box_size
        box_y2 = box_y1 + box_size

        cv2.rectangle(
            legend, (box_x1, box_y1), (box_x2, box_y2), tuple(int(c) for c in color), -1
        )
        cv2.rectangle(
            legend, (box_x1, box_y1), (box_x2, box_y2), (40, 40, 40), 1
        )  # Softer border

        # Draw label text with better positioning
        text_x = box_x2 + max(6, int(8 * scale_factor))

        # Get text size for better vertical centering
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_y = y_offset + item_height // 2 + text_h // 2

        cv2.putText(
            legend,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (40, 40, 40),  # Softer text color
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
    target_size=(256, 256),
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


def _get_spacing_for_scale(scale_factor, component_type="default"):
    """Calculate appropriate spacing based on scale factor and component type."""
    base_spacing = {
        "images": 3,  # Tighter spacing for image rows
        "overlays": 4,  # Medium spacing for overlay rows
        "errors": 3,  # Tighter spacing for error rows
        "default": 4,  # Default spacing
    }
    spacing = base_spacing.get(component_type, 4)
    return max(1, int(spacing * scale_factor))


def _get_legend_width(main_width, scale_factor):
    """Calculate appropriate legend width based on main image width and scale."""
    # Use proportional width with reasonable minimum and maximum
    min_width = max(120, int(120 * scale_factor))
    max_width = min(250, int(250 * scale_factor))
    proportional_width = int(main_width * 0.25)
    return max(min_width, min(proportional_width, max_width))


# ============================================================
# === make_redesigned_panel  (Option A, clean + corrected) ===
# ============================================================


def _create_component(img, title, font_scale=0.6, border_value=(255, 255, 255)):
    """Create a titled image component with a border."""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    component = add_title(img, title, font_scale=font_scale)
    return pad_border(component, value=border_value)


def _create_legend_component(legend_img, title, font_scale=0.6):
    """Create a legend component that doesn't get resized with other components."""
    if legend_img.dtype != np.uint8:
        legend_img = legend_img.astype(np.uint8)

    # Add title but keep the legend at its native size
    component = add_title(legend_img, title, font_scale=font_scale)
    return pad_border(component, value=(255, 255, 255))


def _resize_to_match_height(components, legend_indices=None):
    """Resize a list of images to the maximum height, skipping legends."""
    if legend_indices is None:
        legend_indices = []

    # Find max height excluding legends
    non_legend_heights = [
        c.shape[0] for i, c in enumerate(components) if i not in legend_indices
    ]
    if not non_legend_heights:  # All are legends
        return components

    max_h = max(non_legend_heights)
    resized_components = []
    for i, c in enumerate(components):
        h, w = c.shape[:2]
        if i in legend_indices:
            # Don't resize legends
            resized_components.append(c)
        elif h != max_h:
            new_w = int(w * (max_h / h))
            resized = cv2.resize(c, (new_w, max_h), interpolation=cv2.INTER_AREA)
            resized_components.append(resized)
        else:
            resized_components.append(c)
    return resized_components


def _stack_images(components, axis, spacing=4, bg_color=(255, 255, 255)):
    """Stack images with spacing and a background color."""
    if axis == 0:  # Vertical stacking
        max_w = max(c.shape[1] for c in components)
        padded_components = []
        for c in components:
            h, w = c.shape[:2]
            if w != max_w:
                pad_w = max_w - w
                c = cv2.copyMakeBorder(
                    c, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=bg_color
                )
            padded_components.append(c)

        spaced_components = []
        for i, c in enumerate(padded_components):
            spaced_components.append(c)
            if i < len(padded_components) - 1:
                spacer = np.full((spacing, c.shape[1], 3), bg_color, dtype=np.uint8)
                spaced_components.append(spacer)
        return np.vstack(spaced_components)

    else:  # Horizontal stacking
        max_h = max(c.shape[0] for c in components)
        padded_components = []
        for c in components:
            h, w = c.shape[:2]
            if h != max_h:
                pad_h = max_h - h
                c = cv2.copyMakeBorder(
                    c, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=bg_color
                )
            padded_components.append(c)

        spaced_components = []
        for i, c in enumerate(padded_components):
            spaced_components.append(c)
            if i < len(padded_components) - 1:
                spacer = np.full((c.shape[0], spacing, 3), bg_color, dtype=np.uint8)
                spaced_components.append(spacer)
        return np.hstack(spaced_components)


def _stack_images_with_legends(
    components, legend_indices, axis, spacing=4, bg_color=(255, 255, 255)
):
    """Stack images with legends, handling legends specially to avoid resizing."""
    import glacier_mapping.utils.logging as log

    log.debug(
        f"_stack_images_with_legends: components={len(components)}, legend_indices={legend_indices}"
    )

    if axis == 0:  # Vertical stacking
        return _stack_images(components, axis, spacing, bg_color)

    else:  # Horizontal stacking with special legend handling
        # Separate regular components and legends
        regular_components = [
            c for i, c in enumerate(components) if i not in legend_indices
        ]
        legend_components = [c for i, c in enumerate(components) if i in legend_indices]

        log.debug(
            f"_stack_images_with_legends: regular_components={len(regular_components)}, legend_components={len(legend_components)}"
        )

        if not legend_components:
            return _stack_images(components, axis, spacing, bg_color)

        # Get max height of regular components only
        max_h = (
            max(c.shape[0] for c in regular_components)
            if regular_components
            else max(c.shape[0] for c in legend_components)
        )

        # Process regular components
        padded_regular = []
        for c in regular_components:
            h, w = c.shape[:2]
            if h != max_h:
                pad_h = max_h - h
                c = cv2.copyMakeBorder(
                    c, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=bg_color
                )
            padded_regular.append(c)

    # Process legend components - center them vertically
    padded_legends = []
    log.debug(
        f"_stack_images_with_legends: processing {len(legend_components)} legend components, max_h={max_h}"
    )
    for i, c in enumerate(legend_components):
        h, w = c.shape[:2]
        log.debug(f"_stack_images_with_legends: legend {i} shape=({h}, {w})")
        if h < max_h:
            # Center legend vertically
            pad_top = (max_h - h) // 2
            pad_bottom = max_h - h - pad_top
            c = cv2.copyMakeBorder(
                c, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=bg_color
            )
            log.debug(
                f"_stack_images_with_legends: padded legend {i} to height {max_h}"
            )
        elif h > max_h:
            # Legend is taller than regular components (unlikely but handle it)
            # Resize regular components to match legend height
            max_h = h
            padded_regular = []
            for c_orig in regular_components:
                h_orig, w_orig = c_orig.shape[:2]
                if h_orig != max_h:
                    new_w = int(w_orig * (max_h / h_orig))
                    c_resized = cv2.resize(
                        c_orig, (new_w, max_h), interpolation=cv2.INTER_AREA
                    )
                    padded_regular.append(c_resized)
                else:
                    padded_regular.append(c_orig)
            log.debug(
                f"_stack_images_with_legends: resized regular components to match legend height {max_h}"
            )
        else:
            # Legend height equals max_h, no padding needed
            pass

        # Always append the legend (whether padded or not)
        padded_legends.append(c)
        log.debug(
            f"_stack_images_with_legends: added legend {i} to padded_legends, now {len(padded_legends)} total"
        )

        # Rebuild components list in original order
        final_components = []
        legend_idx = 0
        regular_idx = 0
        log.debug(
            f"_stack_images_with_legends: rebuilding components, padded_legends={len(padded_legends)}, padded_regular={len(padded_regular)}"
        )
        for i in range(len(components)):
            if i in legend_indices:
                log.debug(
                    f"_stack_images_with_legends: i={i}, legend_idx={legend_idx}, padded_legends length={len(padded_legends)}"
                )
                if legend_idx >= len(padded_legends):
                    log.error(
                        f"_stack_images_with_legends: legend_idx {legend_idx} >= len(padded_legends) {len(padded_legends)}"
                    )
                    raise IndexError(
                        f"Legend index out of range: {legend_idx} >= {len(padded_legends)}"
                    )
                final_components.append(padded_legends[legend_idx])
                legend_idx += 1
            else:
                final_components.append(padded_regular[regular_idx])
                regular_idx += 1

        # Add spacing
        spaced_components = []
        for i, c in enumerate(final_components):
            spaced_components.append(c)
            if i < len(final_components) - 1:
                spacer = np.full((c.shape[0], spacing, 3), bg_color, dtype=np.uint8)
                spaced_components.append(spacer)

        return np.hstack(spaced_components)


def make_redesigned_panel(
    context_rgb,
    x_rgb,
    gt_labels,
    pr_labels,
    cmap,
    output_classes=None,
    class_names=None,
    metrics_text=None,
    conf_rgb=None,
    mask=None,
    scale_factor=0.5,
):
    """
    Creates a redesigned, optimized visualization panel for model predictions.

    This function follows a staged pipeline:
    1.  **Single-Pass Scaling**: All inputs are scaled exactly once.
    2.  **Masking & Error Computation**: Mask is applied, and errors (TP/FP/FN) are computed.
    3.  **Component Generation**: All visual assets (overlays, error masks, etc.) are created.
    4.  **Layout Assembly**: Components are arranged into rows with consistent sizing and spacing.
    5.  **Final Composite**: Rows are stacked, and a header is added.
    """
    # --------------------------------------------------------------------------
    # Stage 1: Single-Pass Scaling
    # --------------------------------------------------------------------------
    if scale_factor != 1.0:
        context_rgb = scale_image(context_rgb, scale_factor)
        x_rgb = scale_image(x_rgb, scale_factor)
        gt_labels = scale_image(gt_labels, scale_factor)
        pr_labels = scale_image(pr_labels, scale_factor)
        if conf_rgb is not None:
            conf_rgb = scale_image(conf_rgb, scale_factor)
        if mask is not None:
            mask = scale_image(mask.astype(np.uint8), scale_factor) > 0

    H, W = x_rgb.shape[:2]

    # --------------------------------------------------------------------------
    # Stage 2: Mask Application and Error Computation
    # --------------------------------------------------------------------------
    if mask is not None:
        gt_labels[~mask] = 255
        pr_labels[~mask] = 255

    # Handle binary vs. multiclass error calculation
    if output_classes is None:
        # Default to binary clean ice if not provided
        output_classes = [1]

    tp_mask = np.zeros_like(gt_labels, dtype=bool)
    fp_mask = np.zeros_like(gt_labels, dtype=bool)
    fn_mask = np.zeros_like(gt_labels, dtype=bool)

    for target_class in output_classes:
        if target_class == 0:  # Skip background
            continue

        # For binary models, predictions are always 1 for the target class
        pr_class = 1 if len(output_classes) == 1 else target_class

        tp_mask |= (gt_labels == target_class) & (pr_labels == pr_class)
        fp_mask |= (
            (gt_labels != target_class) & (pr_labels == pr_class) & (gt_labels != 255)
        )
        fn_mask |= (gt_labels == target_class) & (pr_labels != pr_class)

    # --------------------------------------------------------------------------
    # Stage 3: Generate Visual Components
    # --------------------------------------------------------------------------
    gt_rgb = label_to_color(gt_labels, cmap)
    pr_rgb = label_to_color(pr_labels, cmap)

    gt_overlay = make_overlay(x_rgb, gt_labels, cmap, scale_factor=1.0)
    pr_overlay = make_overlay(x_rgb, pr_labels, cmap, scale_factor=1.0)

    errors_overlay = make_errors_overlay(
        x_rgb, tp_mask, fp_mask, fn_mask, scale_factor=1.0
    )
    combined_errors = make_combined_error_mask(tp_mask, fp_mask, fn_mask)
    fp_rgb, fn_rgb = make_fp_fn_individual_masks(fp_mask, fn_mask)

    # --------------------------------------------------------------------------
    # Stage 4: Create Legends
    # --------------------------------------------------------------------------
    from collections import OrderedDict

    class_legend_dict = OrderedDict(
        [
            ("Background", COLOR_BG.tolist()),
            ("Clean Ice", COLOR_CI.tolist()),
            ("Debris", COLOR_DEB.tolist()),
            ("Mask", COLOR_IGNORE.tolist()),
        ]
    )
    # Calculate smart legend width
    legend_width = _get_legend_width(W, scale_factor)
    class_legend = create_legend_row(
        class_legend_dict, width=legend_width, scale_factor=scale_factor
    )

    error_legend_dict = OrderedDict(
        [
            ("True Positive", COLOR_TP.tolist()),
            ("False Positive", COLOR_FP.tolist()),
            ("False Negative", COLOR_FN.tolist()),
        ]
    )
    error_legend = create_legend_row(
        error_legend_dict, width=legend_width, scale_factor=scale_factor
    )

    # --------------------------------------------------------------------------
    # Stage 5: Assemble Panel Rows
    # --------------------------------------------------------------------------
    # Row 1: Context, Slice, Confidence, and Class Legend
    row1_components = [
        _create_component(context_rgb, "Satellite Image (RGB)"),
        _create_component(x_rgb, "Tested Image Slice"),
    ]
    if conf_rgb is not None:
        cbar = make_confidence_colorbar(conf_rgb.shape[1], height=40, font_scale=0.5)
        conf_block = np.vstack([conf_rgb, cbar])
        row1_components.append(_create_component(conf_block, "Confidence"))
    else:
        empty = np.full((H, W, 3), 255, dtype=np.uint8)
        row1_components.append(_create_component(empty, "No Confidence Data"))

    row1_components.append(_create_legend_component(class_legend, "Class Colors"))
    row1_legend_indices = [len(row1_components) - 1]  # Last component is legend
    row1_spacing = _get_spacing_for_scale(scale_factor, "images")
    row1 = _stack_images_with_legends(
        row1_components, row1_legend_indices, axis=1, spacing=row1_spacing
    )

    # Row 2: Overlays and Masks
    row2_components = [
        _create_component(gt_overlay, "Ground Truth Overlay"),
        _create_component(pr_overlay, "Prediction Overlay"),
        _create_component(gt_rgb, "Ground Truth"),
        _create_component(pr_rgb, "Prediction"),
    ]
    row2_spacing = _get_spacing_for_scale(scale_factor, "overlays")
    row2 = _stack_images(
        _resize_to_match_height(row2_components), axis=1, spacing=row2_spacing
    )

    # Row 3: Error Visualizations
    row3_components = [
        _create_component(errors_overlay, "Errors Overlay"),
        _create_component(combined_errors, "Errors (TP+FP+FN)"),
        _create_component(fp_rgb, "False Positives"),
        _create_component(fn_rgb, "False Negatives"),
        _create_legend_component(error_legend, "Error Types"),
    ]
    row3_legend_indices = [len(row3_components) - 1]  # Last component is legend
    row3_spacing = _get_spacing_for_scale(scale_factor, "errors")
    row3 = _stack_images_with_legends(
        row3_components, row3_legend_indices, axis=1, spacing=row3_spacing
    )

    # --------------------------------------------------------------------------
    # Stage 6: Final Assembly
    # --------------------------------------------------------------------------
    vertical_spacing = _get_spacing_for_scale(scale_factor, "default")
    composite = _stack_images([row1, row2, row3], axis=0, spacing=vertical_spacing)

    if metrics_text:
        header_font_scale = max(0.5, 0.6 * scale_factor)
        header = title_bar(
            metrics_text, composite.shape[1], height=40, font_scale=header_font_scale
        )
        composite = np.vstack([header, composite])

    return composite
