"""Unsupervised structure in the enriched clip features.

Rounds 5 and 6 hit a ~0.61 macro-F1 ceiling on DAiSEE engagement. This asks a
different question: ignoring the labels entirely, what natural groupings do
the geometric features form, and do those groupings look anything like
engagement?
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))
from clip_features import build_clip_table, feature_columns  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEQ_PATH = os.path.join(REPO_ROOT, "data", "processed", "daisee_sequences",
                        "validation_sequences.parquet")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

GROUPINGS = {
    2: {0: 0, 1: 0, 2: 0, 3: 1},
    3: {0: 0, 1: 0, 2: 1, 3: 2},
}
K_VALUES = [2, 3, 4]
SEED = 42


def load_features():
    df = pd.read_parquet(SEQ_PATH)
    table = build_clip_table(df)
    cols = feature_columns(table)
    X = StandardScaler().fit_transform(table[cols].to_numpy())
    return table, cols, X


def fit_clusterings(X):
    """K-Means and GMM at each k, scored by silhouette."""
    results = {}
    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        gmm = GaussianMixture(n_components=k, random_state=SEED, covariance_type="full")
        gmm_labels = gmm.fit_predict(X)
        results[k] = {
            "kmeans": {
                "labels": km.labels_,
                "silhouette": float(silhouette_score(X, km.labels_)),
                "inertia": float(km.inertia_),
            },
            "gmm": {
                "labels": gmm_labels,
                "silhouette": float(silhouette_score(X, gmm_labels)),
                "bic": float(gmm.bic(X)),
            },
        }
    return results


def label_alignment(cluster_labels, table):
    """How much does a clustering agree with the engagement labels?"""
    alignment = {}
    for n_classes, mapping in GROUPINGS.items():
        y = table["engagement"].map(mapping).to_numpy()
        alignment[f"{n_classes}_class"] = {
            "adjusted_rand_index": float(adjusted_rand_score(y, cluster_labels)),
            "normalized_mutual_info": float(normalized_mutual_info_score(y, cluster_labels)),
        }
    alignment["raw_4_level"] = {
        "adjusted_rand_index": float(adjusted_rand_score(table["engagement"], cluster_labels)),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(table["engagement"], cluster_labels)),
    }
    return alignment


def characterise_clusters(X, cols, labels):
    """Which features separate the clusters most, in standard deviations."""
    unique = np.unique(labels)
    means = {int(c): X[labels == c].mean(axis=0) for c in unique}
    spread = np.array([max(means[c][i] for c in unique) - min(means[c][i] for c in unique)
                       for i in range(len(cols))])
    order = np.argsort(spread)[::-1]

    top = []
    for i in order[:12]:
        top.append({
            "feature": cols[i],
            "spread_sd": float(spread[i]),
            "cluster_means_sd": {int(c): float(means[c][i]) for c in unique},
        })
    return top, {int(c): int((labels == c).sum()) for c in unique}


def plot_projections(X, table, cluster_labels, k, out_path):
    pca = PCA(n_components=2, random_state=SEED)
    xy_pca = pca.fit_transform(X)
    # PCA down to 30 dims first keeps t-SNE stable on 75 features
    pre = PCA(n_components=30, random_state=SEED).fit_transform(X)
    xy_tsne = TSNE(n_components=2, random_state=SEED, perplexity=30,
                   init="pca").fit_transform(pre)

    engagement2 = table["engagement"].map(GROUPINGS[2]).to_numpy()
    panels = [
        (xy_pca, cluster_labels, f"PCA - k-means clusters (k={k})"),
        (xy_pca, engagement2, "PCA - engagement label (2-class)"),
        (xy_tsne, cluster_labels, f"t-SNE - k-means clusters (k={k})"),
        (xy_tsne, engagement2, "t-SNE - engagement label (2-class)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (xy, colour, title) in zip(axes.ravel(), panels):
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=colour, cmap="viridis", s=6, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(*scatter.legend_elements(), fontsize=7, loc="best")
    var = pca.explained_variance_ratio_
    fig.suptitle(f"clip feature space (PC1+PC2 explain {var[:2].sum():.1%} of variance)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_silhouettes(clusterings, out_path):
    ks = list(clusterings.keys())
    km = [clusterings[k]["kmeans"]["silhouette"] for k in ks]
    gmm = [clusterings[k]["gmm"]["silhouette"] for k in ks]
    x = np.arange(len(ks))
    width = 0.38
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, km, width, label="k-means")
    plt.bar(x + width / 2, gmm, width, label="GMM")
    plt.xticks(x, [f"k={k}" for k in ks])
    plt.ylabel("silhouette score")
    plt.title("cluster separation by k")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table, cols, X = load_features()
    print(f"{len(table)} clips, {len(cols)} features")

    clusterings = fit_clusterings(X)

    print("\n=== silhouette scores ===")
    for k in K_VALUES:
        print(f"k={k}  k-means {clusterings[k]['kmeans']['silhouette']:.4f}   "
              f"GMM {clusterings[k]['gmm']['silhouette']:.4f}")

    best_k = max(K_VALUES, key=lambda k: clusterings[k]["kmeans"]["silhouette"])
    print(f"\nbest k by k-means silhouette: {best_k}")

    results = {"n_clips": len(table), "n_features": len(cols), "best_k": best_k,
               "silhouette": {}, "alignment": {}, "cluster_sizes": {},
               "cluster_characterisation": {}}

    for k in K_VALUES:
        results["silhouette"][f"k={k}"] = {
            "kmeans": clusterings[k]["kmeans"]["silhouette"],
            "gmm": clusterings[k]["gmm"]["silhouette"],
            "kmeans_inertia": clusterings[k]["kmeans"]["inertia"],
            "gmm_bic": clusterings[k]["gmm"]["bic"],
        }
        for algo in ("kmeans", "gmm"):
            labels = clusterings[k][algo]["labels"]
            results["alignment"][f"{algo}_k={k}"] = label_alignment(labels, table)

    print("\n=== alignment with engagement labels ===")
    print(f"{'clustering':16s} {'2-class ARI':>12s} {'2-class NMI':>12s} "
          f"{'3-class ARI':>12s} {'3-class NMI':>12s}")
    for key, align in results["alignment"].items():
        print(f"{key:16s} {align['2_class']['adjusted_rand_index']:12.4f} "
              f"{align['2_class']['normalized_mutual_info']:12.4f} "
              f"{align['3_class']['adjusted_rand_index']:12.4f} "
              f"{align['3_class']['normalized_mutual_info']:12.4f}")

    best_labels = clusterings[best_k]["kmeans"]["labels"]
    top_features, sizes = characterise_clusters(X, cols, best_labels)
    results["cluster_sizes"][f"kmeans_k={best_k}"] = sizes
    results["cluster_characterisation"][f"kmeans_k={best_k}"] = top_features

    print(f"\n=== what separates the k={best_k} k-means clusters ===")
    print(f"cluster sizes: {sizes}")
    for item in top_features:
        means = "  ".join(f"c{c}={v:+.2f}" for c, v in item["cluster_means_sd"].items())
        print(f"  {item['feature']:28s} spread {item['spread_sd']:.2f} sd   {means}")

    # how engagement is distributed inside each cluster
    engagement2 = table["engagement"].map(GROUPINGS[2]).to_numpy()
    print(f"\n=== engagement (2-class) inside each cluster ===")
    crosstab = pd.crosstab(best_labels, engagement2, normalize="index")
    print(crosstab.round(3))
    results["engagement_by_cluster"] = crosstab.to_dict()

    plot_projections(X, table, best_labels, best_k,
                     os.path.join(RESULTS_DIR, "clustering_projections.png"))
    plot_silhouettes(clusterings, os.path.join(RESULTS_DIR, "clustering_silhouette.png"))

    with open(os.path.join(RESULTS_DIR, "clustering_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
