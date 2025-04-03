import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path


def tsne_cluster_misclassifications(features, misclassified_indices, labels_true, output_dir="outputs"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Subset features and labels for misclassified samples
    misclassified_feats = features[misclassified_indices]
    misclassified_labels = np.array(labels_true)[misclassified_indices]

    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    projected = tsne.fit_transform(misclassified_feats)

    best_k = None
    best_score = -1
    best_kmeans = None

    print("\n[Clustering] Testing cluster counts from 2 to 10...")
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(projected)
        score = silhouette_score(projected, labels)
        print(f"  k = {k}, Silhouette Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans

    print(f"\n[Clustering] Best k = {best_k} with Silhouette Score = {best_score:.4f}")

    # Final clustering
    final_labels = best_kmeans.labels_

    # Plot clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(projected[:, 0], projected[:, 1], c=final_labels, cmap='tab10', s=40, alpha=0.8)
    plt.title(f"t-SNE of Misclassified Samples (k={best_k})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "tsne_misclassified_clusters.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Clustering] Saved plot to {save_path}")
