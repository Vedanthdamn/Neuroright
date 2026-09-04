import os
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURES_PATH = os.path.join(REPO_ROOT, "data", "processed", "features.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

FEATURE_COLS = [
    "ear_left", "ear_right", "ear_mean",
    "gaze_left_x", "gaze_right_x", "gaze_x_mean",
    "head_pitch", "head_yaw", "head_roll",
]
LABEL_COL = "label_binary"


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    return X, y, df


def plot_confusion_matrix(cm, title, out_path):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["not engaged", "engaged"],
                yticklabels=["not engaged", "engaged"])
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["not engaged", "engaged"])

    print(f"\n=== {name} ===")
    print(f"accuracy: {acc:.4f}")
    print(f"f1: {f1:.4f}")
    print(report)

    plot_confusion_matrix(cm, f"{name} confusion matrix",
                           os.path.join(RESULTS_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png"))

    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm.tolist(), "report": report}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X, y, df = load_data()
    print(f"loaded {len(df)} samples, {sum(y == 1)} engaged / {sum(y == 0)} not engaged")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_results = evaluate("SVM", svm, X_test_scaled, y_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_results = evaluate("Random Forest", rf, X_test, y_test)

    importances = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))
    print("\nrandom forest feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    summary = {
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "svm": {"accuracy": svm_results["accuracy"], "f1": svm_results["f1"]},
        "random_forest": {"accuracy": rf_results["accuracy"], "f1": rf_results["f1"]},
        "rf_feature_importances": importances,
    }
    with open(os.path.join(RESULTS_DIR, "baseline_svm_rf_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nsaved results and confusion matrices to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
