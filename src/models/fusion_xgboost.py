"""Fusion model: enriched clip aggregates + LSTM embeddings into XGBoost.

Every model here is evaluated on the identical clip-level splits and the same
seeds, so the comparison table is apples to apples.
"""

import json
import os
import sys

# torch and xgboost each load their own libomp on macOS, and the process
# segfaults when both open a parallel region. Pinning OpenMP to one thread
# before either import avoids it; the models here are small enough not to care.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))
from clip_features import build_clip_table, feature_columns  # noqa: E402

import temporal_lstm as tl  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEQ_PATH = os.path.join(REPO_ROOT, "data", "processed", "daisee_sequences", "validation_sequences.parquet")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

LABEL_COLS = ("boredom", "engagement", "confusion", "frustration")
# the 9 frame features round 5 used, before vertical gaze was added
ORIGINAL_FRAME_FEATURES = [
    "ear_left", "ear_right", "ear_mean",
    "gaze_left_x", "gaze_right_x", "gaze_x_mean",
    "head_pitch", "head_yaw", "head_roll",
]
SEEDS = [42, 43, 44, 45, 46]


def split_indices(y, seed):
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed,
                                         stratify=y[temp_idx])
    return train_idx, val_idx, test_idx


def score(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_tree_model(model, X, y, seed):
    train_idx, _, test_idx = split_indices(y, seed)
    model.fit(X[train_idx], y[train_idx])
    return score(y[test_idx], model.predict(X[test_idx]))


def lstm_embeddings(n_classes, seed):
    """Train the LSTM on the train split only, then embed every clip.

    The model never sees val/test labels, so embeddings for those clips are
    not leaking the target into the fusion features.
    """
    run = tl.train(n_classes, seed=seed)
    model, splits = run["_model"], run["_splits"]
    X = torch.tensor(run["_normalised_X"])
    model.eval()
    with torch.no_grad():
        emb = model.embed(X).numpy()
    lstm_metrics = {"accuracy": run["test_accuracy"], "macro_f1": run["test_macro_f1"]}
    return emb, splits, run["_y"], lstm_metrics


def run_grouping(n_classes):
    df = pd.read_parquet(SEQ_PATH)
    table = build_clip_table(df)
    enriched_cols = feature_columns(table)
    original_cols = [f"{c}_{s}" for c in ORIGINAL_FRAME_FEATURES
                     for s in ("mean", "std", "min", "max")]

    y = table["engagement"].map(tl.GROUPINGS[n_classes]["map"]).to_numpy()
    X_enriched = table[enriched_cols].to_numpy()
    X_original = table[original_cols].to_numpy()

    print(f"\n{'=' * 64}\n{n_classes}-class: {len(table)} clips, "
          f"{len(original_cols)} original vs {len(enriched_cols)} enriched features\n{'=' * 64}")

    collected = {name: [] for name in
                 ["majority", "rf_original", "rf_enriched", "lstm_enriched", "fusion"]}
    fusion_cm = None

    for seed in SEEDS:
        train_idx, val_idx, test_idx = split_indices(y, seed)

        majority_class = np.bincount(y[train_idx], minlength=n_classes).argmax()
        collected["majority"].append(
            score(y[test_idx], np.full(len(test_idx), majority_class)))

        collected["rf_original"].append(evaluate_tree_model(
            RandomForestClassifier(n_estimators=300, random_state=seed,
                                   class_weight="balanced", n_jobs=-1),
            X_original, y, seed))
        collected["rf_enriched"].append(evaluate_tree_model(
            RandomForestClassifier(n_estimators=300, random_state=seed,
                                   class_weight="balanced", n_jobs=-1),
            X_enriched, y, seed))

        emb, lstm_splits, y_lstm, lstm_metrics = lstm_embeddings(n_classes, seed)
        collected["lstm_enriched"].append(lstm_metrics)

        # the lstm uses the same seed and stratification, so the splits line up
        assert np.array_equal(lstm_splits["test"], test_idx), "split mismatch"
        assert np.array_equal(y_lstm, y), "label mismatch"

        X_fused = np.hstack([X_enriched, emb])
        counts = np.bincount(y[train_idx], minlength=n_classes)
        sample_weight = (len(train_idx) / (n_classes * counts))[y[train_idx]]

        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, n_jobs=-1,
            eval_metric="mlogloss",
        )
        xgb.fit(X_fused[train_idx], y[train_idx], sample_weight=sample_weight)
        fusion_pred = xgb.predict(X_fused[test_idx])
        collected["fusion"].append(score(y[test_idx], fusion_pred))
        fusion_cm = confusion_matrix(y[test_idx], fusion_pred)

    summary = {}
    for name, runs in collected.items():
        summary[name] = {
            "accuracy": float(np.mean([r["accuracy"] for r in runs])),
            "accuracy_std": float(np.std([r["accuracy"] for r in runs])),
            "macro_f1": float(np.mean([r["macro_f1"] for r in runs])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in runs])),
        }

    summary["_meta"] = {
        "n_clips": len(table),
        "n_original_features": len(original_cols),
        "n_enriched_features": len(enriched_cols),
        "n_embedding_dims": int(emb.shape[1]),
        "seeds": SEEDS,
        "fusion_confusion_matrix": fusion_cm.tolist(),
    }
    return summary


def print_table(n_classes, summary):
    labels = {
        "majority": "majority baseline",
        "rf_original": "RF original aggregates",
        "rf_enriched": "RF enriched aggregates",
        "lstm_enriched": "LSTM enriched sequences",
        "fusion": "fusion (XGBoost)",
    }
    print(f"\n### {n_classes}-class results (mean +/- std over {len(SEEDS)} seeds)")
    print(f"{'model':28s} {'accuracy':>18s} {'macro f1':>18s}")
    for key, label in labels.items():
        s = summary[key]
        print(f"{label:28s} {s['accuracy']:.4f} +/- {s['accuracy_std']:.4f}   "
              f"{s['macro_f1']:.4f} +/- {s['macro_f1_std']:.4f}")


def plot_comparison(results, out_path):
    models = ["majority", "rf_original", "rf_enriched", "lstm_enriched", "fusion"]
    labels = ["majority", "RF orig", "RF enriched", "LSTM", "fusion"]

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5), squeeze=False)
    for ax, (key, summary) in zip(axes[0], results.items()):
        x = np.arange(len(models))
        accs = [summary[m]["accuracy"] for m in models]
        f1s = [summary[m]["macro_f1"] for m in models]
        acc_err = [summary[m]["accuracy_std"] for m in models]
        f1_err = [summary[m]["macro_f1_std"] for m in models]
        width = 0.38
        ax.bar(x - width / 2, accs, width, yerr=acc_err, capsize=3, label="accuracy")
        ax.bar(x + width / 2, f1s, width, yerr=f1_err, capsize=3, label="macro f1")
        ax.axhline(summary["majority"]["accuracy"], ls="--", c="grey", lw=1,
                   label="majority accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{key.replace('_', '-')} engagement")
        ax.set_ylabel("score")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    import argparse

    global SEEDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--classes", type=int, nargs="+", default=[2, 3], choices=[2, 3])
    args = parser.parse_args()

    SEEDS = args.seeds

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    for n_classes in args.classes:
        results[f"{n_classes}_class"] = run_grouping(n_classes)

    for n_classes in args.classes:
        print_table(n_classes, results[f"{n_classes}_class"])

    with open(os.path.join(RESULTS_DIR, "fusion_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    plot_comparison(results, os.path.join(RESULTS_DIR, "model_comparison_fusion.png"))
    print(f"\nsaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
