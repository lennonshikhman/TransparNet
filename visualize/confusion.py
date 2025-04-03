# confusion.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True, title="Confusion Matrix", save_path=None):
    """
    Plots a confusion matrix with optional normalization and class labels.

    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        class_names (list): Optional list of class label strings
        normalize (bool): Whether to normalize rows
        title (str): Title of the plot
        save_path (str): Optional path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
        plt.close()


def compute_kappa(y_true, y_pred):
    """
    Computes Cohen's Kappa Score.

    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels

    Returns:
        float: Kappa score
    """
    return cohen_kappa_score(y_true, y_pred)