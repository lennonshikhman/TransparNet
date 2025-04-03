import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils.helpers import denormalize_tensor
from interpret.gradcam import compute_gradcam
from interpret.guided_backprop import compute_guided_backprop
from interpret.peek import compute_peek_map, compute_peek_overlay
from interpret.composite import plot_composite_grid
import cv2


def get_misclassified_indices(preds, labels):
    return [i for i, (p, t) in enumerate(zip(preds, labels)) if p != t]


def save_misclassified_summary(indices, y_true, y_pred, logits, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary = [
        {
            "index": i,
            "true_label": int(y_true[i]),
            "pred_label": int(y_pred[i]),
            "confidence": float(np.max(logits[i])),
        } for i in indices
    ]
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def visualize_misclassifications(
    model,
    dataset,
    misclassified_indices,
    logits,
    output_dir,
    class_names,
    extractor,
    spatial_extractor,
    pre_layer4,
    device
):
    os.makedirs(output_dir, exist_ok=True)

    for i, idx in enumerate(misclassified_indices):
        image, label = dataset[idx]
        image = image.to(device)
        input_tensor = image.unsqueeze(0)

        # Ensure pred_label is a native Python int
        pred_label = int(logits[idx].argmax())

        # Interpretability maps
        gradcam_map = compute_gradcam(model, input_tensor, target_class=pred_label, target_layer=model.layer4)
        guided_map = compute_guided_backprop(model, input_tensor, target_class=pred_label)

        # PEEK entropy + resized map
        with torch.no_grad():
            spatial_maps = spatial_extractor(pre_layer4(input_tensor)).squeeze().cpu().numpy()
        peek_map = compute_peek_map(spatial_maps)
        peek_overlay = compute_peek_overlay(denormalize_tensor(image), peek_map)

        # Resize PEEK to match Grad-CAM shape
        peek_map_resized = cv2.resize(peek_map, gradcam_map.shape[::-1])
        diff_map = gradcam_map - peek_map_resized
        diff_map = np.clip(diff_map, -1, 1)

        # Save visualization
        composite_path = os.path.join(output_dir, f"misclassified_{idx}.png")
        plot_composite_grid(
            image,
            gradcam_map,
            guided_map,
            peek_overlay,
            diff_map=diff_map,
            pred_label=class_names[pred_label],
            bbox=None,
            save_path=composite_path
        )
