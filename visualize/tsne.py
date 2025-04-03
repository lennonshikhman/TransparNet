# tsne.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(features, labels, class_names=None, title="t-SNE Projection", save_path=None):
    """
    Projects high-dimensional features to 2D using t-SNE and visualizes them.

    Args:
        features (np.ndarray): Feature vectors, shape (N, D)
        labels (np.ndarray): Class labels, shape (N,)
        class_names (list): Optional list of class names for legend
        title (str): Plot title
        save_path (str): Optional path to save the figure
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=labels, cmap="tab10", alpha=0.7, edgecolors='k', linewidths=0.3
    )
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    if class_names:
        legend_labels = np.unique(labels)
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(i)))
                   for i in legend_labels]
        plt.legend(handles, [class_names[i] for i in legend_labels], title="Classes")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    plt.show()