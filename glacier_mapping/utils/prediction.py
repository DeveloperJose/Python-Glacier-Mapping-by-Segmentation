"""Prediction utilities for glacier mapping."""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
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
            current_slice = whole_arr[row:row+window_size[0], column:column+window_size[1], :]
            
            if current_slice.shape[:2] != window_size:
                temp = np.zeros((window_size[0], window_size[1], whole_arr.shape[2]))
                temp[:current_slice.shape[0], :current_slice.shape[1], :] = current_slice
                current_slice = temp
            
            pred = module.predict_slice(current_slice, threshold, preprocess=False, use_mask=False)
            
            # Handle edge cases
            endrow_dest = min(row + window_size[0], y_pred.shape[0])
            endrow_source = min(window_size[0], y_pred.shape[0] - row)
            endcolumn_dest = min(column + window_size[1], y_pred.shape[1])
            endcolumn_source = min(window_size[1], y_pred.shape[1] - column)
            
            y_pred[row:endrow_dest, column:endcolumn_dest] = pred[0:endrow_source, 0:endcolumn_source]
    
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


def softmax_probs(module, x_full):
    """Get probability cube from Lightning module.
    
    Args:
        module: Lightning module with forward method
        x_full: Full image array (H, W, C)
        
    Returns:
        Probability cube (H, W, C)
    """
    use_ch = module.use_channels
    x = x_full[:, :, use_ch]
    x_norm = module.normalize(x)
    
    inp = torch.from_numpy(np.expand_dims(x_norm, 0)).float().to(module.device)
    logits = module.forward(inp.permute(0, 3, 1, 2))
    probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
    return probs


def merge_ci_debris(prob_ci: np.ndarray, prob_deb: np.ndarray, thr_ci: float, thr_deb: float) -> Tuple[np.ndarray, np.ndarray]:
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
    
    ci_mask = binary_fill_holes(prob_ci[:, :, 1] >= thr_ci)
    deb_mask = binary_fill_holes(prob_deb[:, :, 1] >= thr_deb)
    
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