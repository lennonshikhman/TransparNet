# feature_extraction.py

import torch
import numpy as np


def get_feature_extractor(teacher_model):
    """
    Removes the final classification layer from the teacher model to output penultimate features.
    """
    return torch.nn.Sequential(*list(teacher_model.children())[:-1]).eval()


def extract_features(model, dataloader, classifier=None, device='cuda'):
    """
    Extracts features using a CNN model and optionally collects logits from a classifier.

    Args:
        model: CNN feature extractor (usually up to the penultimate layer)
        dataloader: PyTorch DataLoader providing input batches
        classifier: optional full classifier model for logits
        device: target device (e.g., 'cuda' or 'cpu')

    Returns:
        Tuple of (features_array, logits_array)
    """
    features = []
    logits = []
    model = model.to(device).eval()
    if classifier is not None:
        classifier = classifier.to(device).eval()

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            feats = model(images).view(images.size(0), -1)
            features.append(feats.cpu().numpy())

            if classifier:
                with torch.amp.autocast(device_type=device.type):
                    out = classifier(images)
                logits.append(out.cpu().numpy())

    features_np = np.vstack(features)
    logits_np = np.vstack(logits) if logits else None
    return features_np, logits_np
