# guided_backprop.py

import torch
import numpy as np
from captum.attr import GuidedBackprop


def compute_guided_backprop(model, input_tensor, target_class=None, device='cuda'):
    """
    Computes guided backpropagation saliency map.

    Args:
        model (torch.nn.Module): The model to explain.
        input_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W).
        target_class (int, optional): Class index to target. If None, uses model prediction.
        device (str): Device to use (e.g., 'cuda' or 'cpu').

    Returns:
        np.ndarray: 2D saliency map normalized to [0, 1].
    """
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device).requires_grad_(True)

    gbp = GuidedBackprop(model)
    attributions = gbp.attribute(input_tensor, target=target_class)

    saliency = attributions.squeeze().detach().cpu().numpy()  # Shape: (C, H, W)
    saliency = np.mean(np.abs(saliency), axis=0)              # Convert to 2D

    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()

    return saliency
