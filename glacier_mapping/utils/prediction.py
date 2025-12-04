"""Prediction utilities for glacier mapping."""

import numpy as np
import torch
from typing import Tuple
from scipy.ndimage import binary_fill_holes


def predict_whole(module, whole_arr, window_size, threshold=None):
    """Predict on whole image using sliding window.

    Args:
        module: Lightning module with predict_slice method
        whole_arr: Full image array (H, W, C)
        window_size: Tuple of (height, width) for sliding window
        threshold: Prediction threshold (optional)

    Returns:
        Tuple of (prediction_array, mask_array)
    """
    use_channels = module.use_channels
    whole_arr = whole_arr[:, :, use_channels]
    whole_arr = module.normalize(whole_arr)
    mask = np.sum(whole_arr, axis=2) == 0

    y_pred = np.zeros((whole_arr.shape[0], whole_arr.shape[1]), dtype=np.uint8)

    for row in range(0, whole_arr.shape[0], window_size[0]):
        for column in range(0, whole_arr.shape[1], window_size[1]):
            current_slice = whole_arr[
                row : row + window_size[0], column : column + window_size[1], :
            ]

            if current_slice.shape[:2] != window_size:
                temp = np.zeros((window_size[0], window_size[1], whole_arr.shape[2]))
                temp[: current_slice.shape[0], : current_slice.shape[1], :] = (
                    current_slice
                )
                current_slice = temp

            pred = module.predict_slice(
                current_slice, threshold, preprocess=False, use_mask=False
            )

            # Handle edge cases
            endrow_dest = min(row + window_size[0], y_pred.shape[0])
            endrow_source = min(window_size[0], y_pred.shape[0] - row)
            endcolumn_dest = min(column + window_size[1], y_pred.shape[1])
            endcolumn_source = min(window_size[1], y_pred.shape[1] - column)

            y_pred[row:endrow_dest, column:endcolumn_dest] = pred[
                0:endrow_source, 0:endcolumn_source
            ]

    y_pred[mask] = 0
    return y_pred, mask


def get_y_true(label_mask, output_classes, is_binary, binary_class_idx=None, mask=None):
    """Convert one-hot mask to true labels.

    Args:
        label_mask: One-hot encoded mask (H, W, C)
        output_classes: List of output class indices
        is_binary: Whether this is a binary model
        binary_class_idx: Target class index for binary models
        mask: Optional mask to apply

    Returns:
        True labels array (H, W)
    """
    y_true = np.zeros((label_mask.shape[0], label_mask.shape[1]), dtype=np.uint8)

    if is_binary:
        assert binary_class_idx is not None
        y_true[label_mask[:, :, binary_class_idx - 1] != 1] = 0
        y_true[label_mask[:, :, binary_class_idx - 1] == 1] = 1
    else:
        for i in range(label_mask.shape[2]):
            y_true[label_mask[:, :, i] == 1] = i + 1

    if mask is not None:
        y_true[mask] = 0
    return y_true


def get_probabilities(module, x_full):
    """
    Get probability cube from Lightning module using unified softmax approach.

    Args:
        module: Lightning module with forward method
        x_full: Full image array (H, W, C)

    Returns:
        Probability cube (H, W, C) - always uses softmax for consistency
    """
    use_ch = module.use_channels
    x = x_full[:, :, use_ch]
    x_norm = module.normalize(x)

    inp = torch.from_numpy(np.expand_dims(x_norm, 0)).float().to(module.device)
    logits = module.forward(inp.permute(0, 3, 1, 2))
    probs = (
        torch.nn.functional.softmax(logits, dim=1)[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
    return probs


def predict_from_probs(probs, module, threshold=None):
    """
    Convert probabilities to hard predictions using module configuration.

    Args:
        probs: Probability cube (H, W, C) from get_probabilities()
        module: Lightning module for configuration access
        threshold: Optional threshold override (uses module config if None)

    Returns:
        Hard predictions (H, W)
    """
    if len(module.output_classes) == 1:  # Binary
        if threshold is None:
            # Use module's configured threshold
            config_threshold = module.metrics_opts.get("threshold", [0.5])
            threshold = (
                config_threshold[0]
                if isinstance(config_threshold, list)
                else config_threshold
            )
        return (probs[:, :, 1] >= threshold).astype(np.uint8)
    else:  # Multi-class
        return np.argmax(probs, axis=2).astype(np.uint8) + 1


def predict_with_probs(module, x_full, threshold=None):
    """
    Unified prediction function combining probability extraction and hard prediction.

    Args:
        module: Lightning module
        x_full: Full image array (H, W, C)
        threshold: Optional threshold override

    Returns:
        Tuple of (probs, prediction)
        probs: Probability cube (H, W, C)
        prediction: Hard predictions (H, W)
    """
    probs = get_probabilities(module, x_full)
    prediction = predict_from_probs(probs, module, threshold)
    return probs, prediction


def get_pr_iou(pred, true):
    """Calculate precision, recall, IoU, and confusion matrix components.

    Args:
        pred: Predicted labels (numpy array)
        true: Ground truth labels (numpy array)

    Returns:
        Tuple of (precision, recall, IoU, tp, fp, fn)
    """
    from glacier_mapping.model.metrics import tp_fp_fn, precision, recall, IoU

    pred_t = torch.from_numpy(pred.astype(np.uint8))
    true_t = torch.from_numpy(true.astype(np.uint8))
    tp, fp, fn = tp_fp_fn(pred_t, true_t)
    return (precision(tp, fp, fn), recall(tp, fp, fn), IoU(tp, fp, fn), tp, fp, fn)


def calculate_binary_metrics(y_pred, y_true, target_class, mask=None):
    """
    Calculate binary classification metrics using consistent logic.

    Args:
        y_pred: Predicted labels (H, W) - binary output where 1=target, 0=background
        y_true: Ground truth labels (H, W) with original class labels
        target_class: Target class index (1 for CleanIce, 2 for Debris)
        mask: Optional invalid mask where True = invalid pixel

    Returns:
        Tuple of (precision, recall, IoU, tp, fp, fn)
    """
    from glacier_mapping.model.metrics import tp_fp_fn, precision, recall, IoU

    if mask is not None:
        valid = ~mask
        y_pred_valid = y_pred[valid]
        y_true_valid = y_true[valid]
    else:
        y_pred_valid = y_pred
        y_true_valid = y_true

    # Binary conversion: 1=target_class, 0=everything_else
    t_bin = (y_true_valid == target_class).astype(np.uint8)
    p_bin = (y_pred_valid == 1).astype(np.uint8)  # Binary models output 1 for target

    tp_, fp_, fn_ = tp_fp_fn(torch.from_numpy(p_bin), torch.from_numpy(t_bin))
    return (
        precision(tp_, fp_, fn_),
        recall(tp_, fp_, fn_),
        IoU(tp_, fp_, fn_),
        tp_,
        fp_,
        fn_,
    )


def create_invalid_mask(x_full, y_true):
    """
    Create standardized invalid mask for glacier mapping.

    Args:
        x_full: Full image array (H, W, C)
        y_true: Ground truth mask (H, W)

    Returns:
        Invalid mask (H, W) where True = invalid pixel
    """
    return (np.sum(x_full, axis=2) == 0) | (y_true == 255)


# Backward compatibility alias
def softmax_probs(module, x_full):
    """Deprecated: Use get_probabilities() instead."""
    return get_probabilities(module, x_full)


def merge_ci_debris(
    prob_ci: np.ndarray, prob_deb: np.ndarray, thr_ci: float, thr_deb: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine two binary models into a 3-class map.

    Args:
        prob_ci: CleanIce probability cube (H, W, 2)
        prob_deb: Debris probability cube (H, W, 2)
        thr_ci: CleanIce threshold
        thr_deb: Debris threshold

    Returns:
        Tuple of (merged_labels, probability_cube)
        merged_labels: 0=BG, 1=CI, 2=Debris
        probability_cube: 3-class probability cube (H, W, 3)
    """

    # Ensure inputs are valid
    if prob_ci is None or prob_deb is None:
        raise ValueError("Both prob_ci and prob_deb must be provided")

    ci_mask = binary_fill_holes(prob_ci[:, :, 1] >= thr_ci)
    deb_mask = binary_fill_holes(prob_deb[:, :, 1] >= thr_deb)

    # Ensure masks are not None (binary_fill_holes can return None)
    if ci_mask is None:
        ci_mask = prob_ci[:, :, 1] >= thr_ci
    if deb_mask is None:
        deb_mask = prob_deb[:, :, 1] >= thr_deb

    H, W = ci_mask.shape
    merged = np.zeros((H, W), dtype=np.uint8)
    merged[ci_mask] = 1
    merged[deb_mask] = 2

    # probability cube (3-class)
    probs = np.zeros((H, W, 3), dtype=np.float32)
    probs[:, :, 1] = prob_ci[:, :, 1]
    probs[:, :, 2] = prob_deb[:, :, 1]
    probs[:, :, 0] = np.minimum(prob_ci[:, :, 0], prob_deb[:, :, 0])

    return merged, probs
