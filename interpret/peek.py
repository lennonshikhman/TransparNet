# peek.py

import numpy as np
from scipy.special import entr
import cv2


def compute_peek_global(feature_vector):
    """
    Computes global entropy (PEEK score) for a feature vector.
    Args:
        feature_vector (np.ndarray): Shape (C,) or (B, C)
    Returns:
        float or np.ndarray: Entropy score(s)
    """
    x = feature_vector + np.abs(np.min(feature_vector)) + 1e-8
    return -np.sum(entr(x), axis=-1)


def compute_peek_map(spatial_maps):
    """
    Computes a spatial entropy map (H, W) from a feature tensor (C, H, W).
    Args:
        spatial_maps (np.ndarray): Channels-first activation map
    Returns:
        np.ndarray: Normalized entropy map (H, W)
    """
    C, H, W = spatial_maps.shape
    entropy_map = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            patch = spatial_maps[:, i, j]
            patch = patch + np.abs(np.min(patch)) + 1e-8
            entropy_map[i, j] = -np.sum(entr(patch))

    entropy_map -= entropy_map.min()
    if entropy_map.max() > 0:
        entropy_map /= entropy_map.max()
    return entropy_map


def compute_peek_overlay(original_img, peek_map, alpha=0.5):
    """
    Overlays a heatmap on the original image. Assumes:
    - original_img: (H, W, 3) normalized to [0, 1]
    - peek_map: (H', W') or (H', W', 3)
    """
    import cv2

    # Ensure peek_map is 2D before applying colormap
    if peek_map.ndim == 2:
        peek_map = (peek_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(peek_map, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
    else:
        heatmap = peek_map  # Already colored and normalized

    # Resize to match original image size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Blend
    overlay = np.clip(original_img * (1 - alpha) + heatmap * alpha, 0, 1)
    return overlay



def compute_peek_hw_map(feature_tensor, target_size):
    """
    Computes a resized PEEK entropy map for overlay, from a 3D feature tensor.
    Args:
        feature_tensor (np.ndarray): (C, h, w)
        target_size (tuple): Desired output size (H, W)
    Returns:
        np.ndarray: Resized normalized entropy map
    """
    peek_map = compute_peek_map(feature_tensor)
    resized = cv2.resize(peek_map, target_size, interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0, 1)
