import os
import cv2
import pandas as pd
from tqdm import tqdm

from face_features import FaceFeatureExtractor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
OUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features.csv")

# (dataset name, root dir, path -> (label_binary, label_raw, extra_meta) )
KAGGLE_ROOT = os.path.join(RAW_DIR, "kaggle_studentengagement")
ZENODO_ROOT = os.path.join(RAW_DIR, "zenodo_engagement", "Final Dataset 256")

KAGGLE_ENGAGED = {"engaged", "confused", "frustrated"}
KAGGLE_DISENGAGED = {"bored", "drowsy", "Looking Away"}


def iter_kaggle():
    for top, label_bin in (("Engaged", 1), ("Not engaged", 0)):
        top_dir = os.path.join(KAGGLE_ROOT, top)
        if not os.path.isdir(top_dir):
            continue
        for subclass in os.listdir(top_dir):
            sub_dir = os.path.join(top_dir, subclass)
            if not os.path.isdir(sub_dir):
                continue
            for fname in os.listdir(sub_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                yield {
                    "image_path": os.path.join(sub_dir, fname),
                    "dataset": "kaggle_studentengagement",
                    "label_binary": label_bin,
                    "label_raw": subclass,
                    "age_group": None,
                }


def iter_zenodo():
    for top, label_bin in (("Engagement", 1), ("Disengagement", 0)):
        top_dir = os.path.join(ZENODO_ROOT, top)
        if not os.path.isdir(top_dir):
            continue
        for age_group in os.listdir(top_dir):
            age_dir = os.path.join(top_dir, age_group)
            if not os.path.isdir(age_dir):
                continue
            for fname in os.listdir(age_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                yield {
                    "image_path": os.path.join(age_dir, fname),
                    "dataset": "zenodo_engagement",
                    "label_binary": label_bin,
                    "label_raw": top,
                    "age_group": age_group,
                }


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    extractor = FaceFeatureExtractor()

    samples = list(iter_kaggle()) + list(iter_zenodo())
    print(f"found {len(samples)} images total")

    rows = []
    n_no_face = 0
    for sample in tqdm(samples):
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue
        features = extractor.extract(img)
        if features is None:
            n_no_face += 1
            continue
        rows.append({**sample, **features})

    extractor.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(df)} rows to {OUT_PATH}")
    print(f"skipped {n_no_face} images with no face detected")
    print(df["dataset"].value_counts())
    print(df["label_binary"].value_counts())


if __name__ == "__main__":
    main()
