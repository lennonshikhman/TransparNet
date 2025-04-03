# decision_tree_plot.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree


def plot_decision_tree(model, feature_names=None, class_names=None, title="Decision Tree Visualization", save_path=None):
    """
    Visualizes a trained decision tree classifier.

    Args:
        model (DecisionTreeClassifier): Trained decision tree
        feature_names (list): Optional list of feature names (e.g. PCA or visual descriptors)
        class_names (list): Optional list of class label names
        title (str): Title of the plot
        save_path (str): Optional file path to save the visualization
    """
    plt.figure(figsize=(25, 15))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=10
    )
    plt.title(title, fontsize=16)
    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        print(f"Saved decision tree visualization to {save_path}")
    plt.show()


def get_used_features(tree_model):
    """
    Returns sorted unique indices of features used in the tree (ignores leaf nodes).

    Args:
        tree_model (DecisionTreeClassifier): Trained decision tree

    Returns:
        list[int]: Unique feature indices
    """
    indices = tree_model.tree_.feature
    return sorted(np.unique(indices[indices >= 0]))