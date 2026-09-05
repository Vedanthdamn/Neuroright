"""Fit the reference models on the full DAiSEE validation set and persist them.

Rounds 5-7 trained everything inline per seed, which is fine for reporting
scores but leaves nothing to score new data with. This fits one copy of each
on all 1429 clips and saves it, so the demo can reuse the exact same
scaler, cluster assignment and classifiers.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))
from clip_features import build_clip_table, feature_columns  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEQ_PATH = os.path.join(REPO_ROOT, "data", "processed", "daisee_sequences",
                        "validation_sequences.parquet")
MODEL_DIR = os.path.join(REPO_ROOT, "models")

GROUPINGS = {2: {0: 0, 1: 0, 2: 0, 3: 1}, 3: {0: 0, 1: 0, 2: 1, 3: 2}}
SEED = 42
N_CLUSTERS = 2


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_parquet(SEQ_PATH)
    table = build_clip_table(df)
    cols = feature_columns(table)
    X_raw = table[cols].to_numpy()

    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    print(f"fitted scaler on {len(table)} clips, {len(cols)} features")

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10).fit(X)
    sizes = np.bincount(kmeans.labels_, minlength=N_CLUSTERS)
    print(f"kmeans k={N_CLUSTERS} cluster sizes: {sizes.tolist()}")

    # label the clusters by how much the subject moves, so the demo can talk
    # about a "movement" and a "stillness" cluster rather than 0 and 1
    motion_idx = cols.index("head_motion_magnitude")
    cluster_motion = {c: float(X[kmeans.labels_ == c][:, motion_idx].mean())
                      for c in range(N_CLUSTERS)}
    movement_cluster = int(max(cluster_motion, key=cluster_motion.get))
    stillness_cluster = int(min(cluster_motion, key=cluster_motion.get))
    print(f"movement cluster: {movement_cluster}, stillness cluster: {stillness_cluster}")
    print(f"mean scaled head_motion_magnitude per cluster: {cluster_motion}")

    classifiers = {}
    for n_classes, mapping in GROUPINGS.items():
        y = table["engagement"].map(mapping).to_numpy()
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                    class_weight="balanced", n_jobs=-1)
        rf.fit(X_raw, y)
        classifiers[n_classes] = rf
        print(f"fitted {n_classes}-class RF on all clips")

    joblib.dump({
        "scaler": scaler,
        "kmeans": kmeans,
        "feature_columns": cols,
        "movement_cluster": movement_cluster,
        "stillness_cluster": stillness_cluster,
        "classifiers": classifiers,
    }, os.path.join(MODEL_DIR, "reference_models.joblib"))

    meta = {
        "n_clips_fitted": len(table),
        "n_features": len(cols),
        "n_clusters": N_CLUSTERS,
        "cluster_sizes": sizes.tolist(),
        "movement_cluster": movement_cluster,
        "stillness_cluster": stillness_cluster,
        "cluster_mean_scaled_head_motion": cluster_motion,
    }
    with open(os.path.join(MODEL_DIR, "reference_models_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nsaved reference models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
