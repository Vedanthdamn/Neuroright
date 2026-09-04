import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEQ_PATH = os.path.join(REPO_ROOT, "data", "processed", "daisee_sequences", "validation_sequences.parquet")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

FEATURE_COLS = [
    "ear_left", "ear_right", "ear_mean",
    "gaze_left_x", "gaze_right_x", "gaze_x_mean",
    "head_pitch", "head_yaw", "head_roll",
]
SEQ_LEN = 50

# engagement 0-3 collapsed into usable class balance
GROUPINGS = {
    2: {"map": {0: 0, 1: 0, 2: 0, 3: 1}, "names": ["not high (0-2)", "high (3)"]},
    3: {"map": {0: 0, 1: 0, 2: 1, 3: 2}, "names": ["low (0-1)", "medium (2)", "high (3)"]},
}


def build_sequences(n_classes):
    df = pd.read_parquet(SEQ_PATH)
    df = df.sort_values(["clip_id", "frame_idx"])
    label_map = GROUPINGS[n_classes]["map"]

    X, y, clip_ids = [], [], []
    for clip_id, group in df.groupby("clip_id"):
        feats = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        # a handful of clips run 46-49 frames; repeat the last frame to pad
        if len(feats) < SEQ_LEN:
            pad = np.repeat(feats[-1:], SEQ_LEN - len(feats), axis=0)
            feats = np.concatenate([feats, pad], axis=0)
        X.append(feats[:SEQ_LEN])
        y.append(label_map[int(group["engagement"].iloc[0])])
        clip_ids.append(clip_id)

    return np.stack(X), np.array(y), np.array(clip_ids)


class EngagementLSTM(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.fc(self.dropout(pooled))


def augment(batch, noise_std=0.05, jitter=2):
    """light augmentation: gaussian noise plus a small circular time shift"""
    noisy = batch + torch.randn_like(batch) * noise_std
    shift = int(torch.randint(-jitter, jitter + 1, (1,)).item())
    return torch.roll(noisy, shifts=shift, dims=1)


def aggregate_rf_baseline(n_classes, seed=42):
    """Random forest on per-clip summary stats. Same task and split as the LSTM,
    so it shows whether sequence modelling buys anything over flat aggregates."""
    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_parquet(SEQ_PATH)
    agg = df.groupby("clip_id")[FEATURE_COLS].agg(["mean", "std", "min", "max"]).fillna(0)
    agg.columns = ["_".join(c) for c in agg.columns]
    labels = df.groupby("clip_id")["engagement"].first().map(GROUPINGS[n_classes]["map"])

    X, y = agg.to_numpy(), labels.to_numpy()
    idx = np.arange(len(X))
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    _, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, stratify=y[temp_idx])

    rf = RandomForestClassifier(n_estimators=300, random_state=seed,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X[train_idx], y[train_idx])
    pred = rf.predict(X[test_idx])
    return {
        "test_accuracy": accuracy_score(y[test_idx], pred),
        "test_macro_f1": f1_score(y[test_idx], pred, average="macro"),
    }


def plot_comparison(results, out_path):
    labels, accs, f1s = [], [], []
    for key, res in results.items():
        n = res["n_classes"]
        labels += [f"{n}-class\nmajority", f"{n}-class\nRF aggregates", f"{n}-class\nLSTM"]
        accs += [res["majority_baseline_accuracy"], res["rf_aggregate_baseline"]["test_accuracy"],
                 res["test_accuracy"]]
        f1s += [res["majority_macro_f1"], res["rf_aggregate_baseline"]["test_macro_f1"],
                res["test_macro_f1"]]

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(10, 4.5))
    plt.bar(x - width / 2, accs, width, label="accuracy")
    plt.bar(x + width / 2, f1s, width, label="macro f1")
    plt.xticks(x, labels, fontsize=8)
    plt.ylabel("score")
    plt.title("DAiSEE engagement: LSTM vs flat-aggregate RF vs majority baseline")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_confusion_matrix(cm, names, title, out_path):
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train(n_classes, epochs=120, patience=15, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    names = GROUPINGS[n_classes]["names"]
    X, y, clip_ids = build_sequences(n_classes)
    print(f"{len(X)} sequences, shape {X.shape}")
    for i, name in enumerate(names):
        n = int((y == i).sum())
        print(f"  {name}: {n} ({n / len(y) * 100:.1f}%)")

    # split by clip so no frames from one clip land in two splits
    idx = np.arange(len(X))
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, stratify=y[temp_idx])
    print(f"\nsplit: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    # standardise using train statistics only
    mean = X[train_idx].reshape(-1, len(FEATURE_COLS)).mean(axis=0)
    std = X[train_idx].reshape(-1, len(FEATURE_COLS)).std(axis=0) + 1e-8
    Xn = (X - mean) / std

    X_train = torch.tensor(Xn[train_idx])
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(Xn[val_idx])
    y_val = torch.tensor(y[val_idx], dtype=torch.long)
    X_test = torch.tensor(Xn[test_idx])
    y_test = torch.tensor(y[test_idx], dtype=torch.long)

    counts = np.bincount(y[train_idx], minlength=n_classes)
    class_weights = torch.tensor(len(train_idx) / (n_classes * counts), dtype=torch.float32)
    print(f"class weights: {class_weights.tolist()}")

    model = EngagementLSTM(len(FEATURE_COLS), n_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None
    epochs_without_gain = 0
    history = []
    batch_size = 32

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start:start + batch_size]
            xb = augment(X_train[batch_idx])
            yb = y_train[batch_idx]
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch_idx)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).argmax(dim=1)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        val_acc = accuracy_score(y_val, val_pred)
        history.append({"epoch": epoch, "train_loss": total_loss / len(X_train),
                        "val_acc": val_acc, "val_macro_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1

        if epoch % 10 == 0 or epochs_without_gain == 0:
            print(f"epoch {epoch:3d}  loss {total_loss / len(X_train):.4f}  "
                  f"val acc {val_acc:.4f}  val macro-f1 {val_f1:.4f}")

        if epochs_without_gain >= patience:
            print(f"early stopping at epoch {epoch} (best val macro-f1 {best_val_f1:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(dim=1)

    acc = accuracy_score(y_test, test_pred)
    macro_f1 = f1_score(y_test, test_pred, average="macro")
    cm = confusion_matrix(y_test, test_pred)
    majority_class = int(np.bincount(y[test_idx], minlength=n_classes).argmax())
    majority = float(np.bincount(y[test_idx], minlength=n_classes).max() / len(test_idx))
    majority_pred = np.full(len(test_idx), majority_class)
    majority_f1 = f1_score(y[test_idx], majority_pred, average="macro")

    rf_baseline = aggregate_rf_baseline(n_classes, seed=seed)

    print(f"\n=== LSTM ({n_classes}-class) test results ===")
    print(f"accuracy:  {acc:.4f}")
    print(f"macro f1:  {macro_f1:.4f}")
    print(f"majority-class baseline: acc {majority:.4f}, macro f1 {majority_f1:.4f}")
    print(f"RF on clip aggregates:   acc {rf_baseline['test_accuracy']:.4f}, "
          f"macro f1 {rf_baseline['test_macro_f1']:.4f}")
    print(classification_report(y_test, test_pred, target_names=names, zero_division=0))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_confusion_matrix(cm, names, f"LSTM {n_classes}-class confusion matrix",
                          os.path.join(RESULTS_DIR, f"confusion_matrix_lstm_{n_classes}class.png"))

    return {
        "n_classes": n_classes,
        "class_names": names,
        "class_distribution": {names[i]: int((y == i).sum()) for i in range(n_classes)},
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "majority_baseline_accuracy": majority,
        "majority_macro_f1": majority_f1,
        "rf_aggregate_baseline": rf_baseline,
        "best_val_macro_f1": best_val_f1,
        "epochs_run": len(history),
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_test, test_pred, target_names=names,
                                        zero_division=0, output_dict=True),
    }


def run_seeds(n_classes, seeds):
    """The test split is only ~143 clips, so a single run swings several points.
    Average over seeds to get a number worth reporting."""
    runs = [train(n_classes, seed=s) for s in seeds]

    summary = dict(runs[-1])
    for metric in ("test_accuracy", "test_macro_f1"):
        values = [r[metric] for r in runs]
        summary[metric] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
        summary[f"{metric}_per_seed"] = values
    for metric in ("test_accuracy", "test_macro_f1"):
        values = [r["rf_aggregate_baseline"][metric] for r in runs]
        summary["rf_aggregate_baseline"][metric] = float(np.mean(values))
        summary["rf_aggregate_baseline"][f"{metric}_std"] = float(np.std(values))
    summary["seeds"] = seeds
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, choices=[2, 3], default=2)
    parser.add_argument("--both", action="store_true", help="run both groupings")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    targets = [2, 3] if args.both else [args.classes]
    results = {}
    for n in targets:
        print(f"\n{'=' * 60}\ntraining {n}-class model over {len(args.seeds)} seeds\n{'=' * 60}")
        results[f"{n}_class"] = run_seeds(n, args.seeds)
        r = results[f"{n}_class"]
        print(f"\n>>> {n}-class over seeds {args.seeds}")
        print(f"    LSTM acc {r['test_accuracy']:.4f} +/- {r['test_accuracy_std']:.4f}, "
              f"macro f1 {r['test_macro_f1']:.4f} +/- {r['test_macro_f1_std']:.4f}")
        print(f"    RF   acc {r['rf_aggregate_baseline']['test_accuracy']:.4f}, "
              f"macro f1 {r['rf_aggregate_baseline']['test_macro_f1']:.4f}")

    with open(os.path.join(RESULTS_DIR, "temporal_lstm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    plot_comparison(results, os.path.join(RESULTS_DIR, "model_comparison_lstm.png"))
    print(f"\nsaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
