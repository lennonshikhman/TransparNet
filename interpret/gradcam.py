# gradcam.py

import torch
import numpy as np
import cv2


def compute_gradcam(model, input_tensor, target_class=None, target_layer=None, device='cuda'):
    """
    Computes a Grad-CAM heatmap for a given input image and target layer.

    Args:
        model (torch.nn.Module): Full model.
        input_tensor (torch.Tensor): Image tensor, shape (1, C, H, W).
        target_class (int, optional): Class index for which to compute Grad-CAM.
        target_layer (torch.nn.Module): Layer to hook into for gradients and activations.
        device (str): Target device.

    Returns:
        np.ndarray: Heatmap as a 2D array normalized to [0, 1].
    """
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device).requires_grad_(True)

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()

    # Postprocess
    grad = gradients[0].detach().cpu().numpy()[0]     # [C, H, W]
    act = activations[0].detach().cpu().numpy()[0]    # [C, H, W]

    weights = np.mean(grad, axis=(1, 2))              # [C]
    cam = np.zeros_like(act[0])
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    # Resize to input image size
    _, _, H, W = input_tensor.shape
    cam = cv2.resize(cam, (W, H))

    # Cleanup
    forward_handle.remove()
    backward_handle.remove()

    return cam