# composite.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.helpers import denormalize_tensor


def plot_composite_grid(
    image_tensor,
    gradcam_map,
    guided_backprop_map,
    peek_overlay,
    diff_map,
    pred_label,
    bbox=None,
    save_path=None
):
    """
    Creates a 2x3 panel of composite interpretability visuals.

    Args:
        image_tensor (torch.Tensor): Normalized input image (C, H, W)
        gradcam_map (np.ndarray): Grad-CAM heatmap (H, W)
        guided_backprop_map (np.ndarray): Guided BP map (H, W)
        peek_overlay (np.ndarray): RGB overlay image (H, W, 3)
        diff_map (np.ndarray): Difference map (H, W)
        pred_label (str): Predicted class name
        bbox (tuple): Optional (x1, y1, x2, y2) to draw
        save_path (str): Optional path to save the image

    Returns:
        None
    """
    disp_img = denormalize_tensor(image_tensor)
    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        disp_img = cv2.rectangle(
            (disp_img * 255).astype(np.uint8),
            (x1, y1), (x2, y2),
            color=(0, 255, 0),
            thickness=2
        ).astype(np.float32) / 255.0

    gradcam_colored = cv2.applyColorMap((gradcam_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    gradcam_colored = cv2.cvtColor(gradcam_colored, cv2.COLOR_BGR2RGB) / 255.0
    gradcam_overlay = np.clip(disp_img * 0.6 + gradcam_colored * 0.4, 0, 1)

    composite_overlay = np.clip(disp_img * 0.6 + gradcam_map[..., None] * 0.2 + guided_backprop_map[..., None] * 0.2, 0, 1)

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    axs[0].imshow(disp_img)
    axs[0].set_title(f"Original (Pred: {pred_label})")

    axs[1].imshow(gradcam_overlay)
    axs[1].set_title("Grad-CAM Overlay")

    axs[2].imshow(guided_backprop_map, cmap='hot')
    axs[2].set_title("Guided Backprop")

    axs[3].imshow(peek_overlay)
    axs[3].set_title("PEEK Overlay")

    axs[4].imshow(diff_map, cmap='bwr')
    axs[4].set_title("Difference (GradCAM - PEEK)")

    axs[5].imshow(composite_overlay)
    axs[5].set_title("Composite Overlay")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved composite visualization to {save_path}")
    plt.show()