# helpers.py

import numpy as np
import cv2
import torch


def format_elapsed_time(seconds):
    """
    Converts elapsed time in seconds to a human-readable format.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    parts = []
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    if s: parts.append(f"{s}s")
    if ms and not parts: parts.append(f"{ms}ms")
    return ' '.join(parts)


def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Reverses ImageNet normalization for display.

    Args:
        tensor (torch.Tensor): Image tensor (C, H, W)
        mean (list): Mean used in normalization
        std (list): Std used in normalization

    Returns:
        np.ndarray: Image in (H, W, C), float32, values in [0, 1]
    """
    img = tensor.clone().detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)


def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box on an RGB image.

    Args:
        image (np.ndarray): Image in (H, W, 3) with values in [0, 1] or [0, 255]
        bbox (tuple): (x1, y1, x2, y2)
        color (tuple): RGB color
        thickness (int): Thickness of box

    Returns:
        np.ndarray: Image with box (float32, [0, 1])
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    x1, y1, x2, y2 = map(int, bbox)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image.astype(np.float32) / 255.0


def difference_map(map1, map2):
    """
    Computes a normalized difference between two maps.
    
    Args:
        map1 (np.ndarray): First map
        map2 (np.ndarray): Second map

    Returns:
        np.ndarray: Normalized difference map in [0, 1]
    """
    diff = map1 - map2
    diff_min, diff_max = diff.min(), diff.max()
    if diff_max - diff_min > 0:
        diff = (diff - diff_min) / (diff_max - diff_min)
    return diff
