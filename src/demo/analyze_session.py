"""Score self-collected clips against the models fitted on DAiSEE.

Two questions, deliberately separate:

1. Does the movement-vs-stillness axis found in round 7 generalise to a new
   subject? Tested by assigning each clip to the round 7 k-means cluster and
   checking it against the deliberate state label.
2. What do the round 5/6 engagement classifiers say? Reported for
   completeness only; rounds 5-7 established they are at a ceiling.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))
from clip_features import clip_features  # noqa: E402
from face_features import FaceFeatureExtractor  # noqa: E402
from video_sequences import extract_sequence  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SELF_ROOT = os.path.join(REPO_ROOT, "data", "self_collected")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "reference_models.joblib")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# which cluster each deliberate state is expected to fall into
EXPECTED_CLUSTER = {
    "still_attentive": "stillness",
    "fidgety_distracted": "movement",
    "drowsy": "movement",
}
MIN_FRAMES = 10


def collect_clips():
    clips = []
    for state in sorted(EXPECTED_CLUSTER):
        state_dir = os.path.join(SELF_ROOT, state)
        if not os.path.isdir(state_dir):
            continue
        for fname in sorted(os.listdir(state_dir)):
            if fname.lower().endswith((".mp4", ".avi")):
                clips.append({"state": state, "path": os.path.join(state_dir, fname)})
    return clips


def build_table(clips, feature_columns):
    extractor = FaceFeatureExtractor()
    rows, meta, skipped = [], [], []

    for clip in clips:
        seq, n_sampled = extract_sequence(clip["path"], extractor=extractor)
        if len(seq) < MIN_FRAMES:
            skipped.append({"path": clip["path"], "state": clip["state"],
                            "frames_with_face": len(seq), "frames_sampled": n_sampled})
            continue
        feats = clip_features(seq)
        rows.append([feats[c] for c in feature_columns])
        meta.append({"path": clip["path"], "state": clip["state"],
                     "frames_with_face": len(seq), "frames_sampled": n_sampled})

    extractor.close()
    X = np.array(rows, dtype=float) if rows else np.empty((0, len(feature_columns)))
    return X, pd.DataFrame(meta), skipped


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("reference models missing - run src/models/fit_reference_models.py first")

    bundle = joblib.load(MODEL_PATH)
    cols = bundle["feature_columns"]
    cluster_name = {bundle["movement_cluster"]: "movement",
                    bundle["stillness_cluster"]: "stillness"}

    clips = collect_clips()
    if not clips:
        raise SystemExit(
            f"no clips found under {SELF_ROOT}. record some with "
            "src/demo/record_session.py first"
        )
    print(f"found {len(clips)} clips")

    X, meta, skipped = build_table(clips, cols)
    if len(X) == 0:
        raise SystemExit("no clip had enough frames with a detected face")

    if skipped:
        print(f"\nskipped {len(skipped)} clips with under {MIN_FRAMES} usable frames:")
        for s in skipped:
            print(f"  {os.path.basename(s['path'])}: {s['frames_with_face']}"
                  f"/{s['frames_sampled']} frames had a face")

    X_scaled = bundle["scaler"].transform(X)
    meta["cluster"] = bundle["kmeans"].predict(X_scaled)
    meta["cluster_name"] = meta["cluster"].map(cluster_name)
    meta["expected_cluster"] = meta["state"].map(EXPECTED_CLUSTER)
    meta["cluster_matches_expected"] = meta["cluster_name"] == meta["expected_cluster"]

    print(f"\n=== clips analysed: {len(meta)} ===")
    print(meta.groupby("state").size().to_string())

    print("\n=== cluster assignment by deliberate state ===")
    crosstab = pd.crosstab(meta["state"], meta["cluster_name"])
    print(crosstab.to_string())
    print("\nrow-normalised:")
    print(pd.crosstab(meta["state"], meta["cluster_name"], normalize="index").round(3).to_string())

    agreement = float(meta["cluster_matches_expected"].mean())
    print(f"\ncluster matches expected state grouping: {agreement:.1%} "
          f"({int(meta['cluster_matches_expected'].sum())}/{len(meta)} clips)")

    # still vs (fidgety + drowsy) is the binary the clustering can actually speak to
    still_mask = meta["state"] == "still_attentive"
    still_rate = float((meta.loc[still_mask, "cluster_name"] == "stillness").mean()) \
        if still_mask.any() else float("nan")
    active_rate = float((meta.loc[~still_mask, "cluster_name"] == "movement").mean()) \
        if (~still_mask).any() else float("nan")
    print(f"  still_attentive -> stillness cluster: {still_rate:.1%}")
    print(f"  fidgety/drowsy  -> movement cluster:  {active_rate:.1%}")

    classifier_out = {}
    print("\n=== round 5/6 engagement classifiers (context only) ===")
    for n_classes, clf in bundle["classifiers"].items():
        preds = clf.predict(X)
        dist = pd.Series(preds).value_counts(normalize=True).sort_index().round(3).to_dict()
        by_state = pd.crosstab(meta["state"], preds, normalize="index").round(3)
        print(f"\n{n_classes}-class predicted label distribution: "
              f"{ {int(k): float(v) for k, v in dist.items()} }")
        print(by_state.to_string())
        classifier_out[f"{n_classes}_class"] = {
            "predicted_distribution": {int(k): float(v) for k, v in dist.items()},
            "by_state": by_state.to_dict(),
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "n_clips_found": len(clips),
        "n_clips_analysed": int(len(meta)),
        "n_clips_skipped": len(skipped),
        "skipped": skipped,
        "clips_per_state": meta.groupby("state").size().to_dict(),
        "cluster_crosstab": crosstab.to_dict(),
        "cluster_agreement_with_expected": agreement,
        "still_to_stillness_rate": still_rate,
        "active_to_movement_rate": active_rate,
        "classifiers": classifier_out,
    }
    with open(os.path.join(RESULTS_DIR, "self_collected_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta.drop(columns=["path"]).to_csv(
        os.path.join(RESULTS_DIR, "self_collected_clip_assignments.csv"), index=False)
    print(f"\nsaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
