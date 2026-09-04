import os

import cv2
import pandas as pd
from tqdm import tqdm

from face_features import FaceFeatureExtractor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VALIDATION_DIR = os.path.join(REPO_ROOT, "data", "raw", "daisee", "validation")
LABELS_PATH = os.path.join(REPO_ROOT, "data", "raw", "daisee", "labels", "ValidationLabels.csv")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "daisee_sequences")
OUT_PATH = os.path.join(OUT_DIR, "validation_sequences.parquet")

SAMPLE_FPS = 5


def build_clip_index():
    """map clip filename (e.g. '4000221001.avi') -> full path on disk"""
    index = {}
    for root, _, fnames in os.walk(VALIDATION_DIR):
        for fname in fnames:
            if fname.lower().endswith((".avi", ".mp4")):
                index[fname] = os.path.join(root, fname)
    return index


def sample_frames(video_path, sample_fps):
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native_fps / sample_fps))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append((frame_idx, frame))
        frame_idx += 1
    cap.release()
    return frames


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    labels = pd.read_csv(LABELS_PATH)
    labels.columns = [c.strip() for c in labels.columns]
    clip_index = build_clip_index()

    print(f"{len(labels)} labeled clips, {len(clip_index)} video files found on disk")

    extractor = FaceFeatureExtractor()
    rows = []
    clips_ok = 0
    clips_no_video = 0
    clips_no_frames_detected = 0
    total_frames_sampled = 0
    total_frames_detected = 0

    for _, label_row in tqdm(labels.iterrows(), total=len(labels)):
        clip_id = label_row["ClipID"]
        video_path = clip_index.get(clip_id)
        if video_path is None:
            clips_no_video += 1
            continue

        frames = sample_frames(video_path, SAMPLE_FPS)
        total_frames_sampled += len(frames)

        clip_had_detection = False
        for frame_idx, frame in frames:
            features = extractor.extract(frame)
            if features is None:
                continue
            clip_had_detection = True
            total_frames_detected += 1
            rows.append({
                "clip_id": clip_id,
                "frame_idx": frame_idx,
                "boredom": label_row["Boredom"],
                "engagement": label_row["Engagement"],
                "confusion": label_row["Confusion"],
                "frustration": label_row["Frustration"],
                **features,
            })

        if clip_had_detection:
            clips_ok += 1
        else:
            clips_no_frames_detected += 1

    extractor.close()

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nclips with label but no video file found: {clips_no_video}")
    print(f"clips processed with at least one detected face: {clips_ok}")
    print(f"clips with zero frames detected (fully dropped): {clips_no_frames_detected}")
    print(f"total frames sampled: {total_frames_sampled}")
    print(f"total frames with a detected face: {total_frames_detected}")
    print(f"frame-level detection rate: {total_frames_detected / total_frames_sampled:.2%}" if total_frames_sampled else "n/a")
    print(f"\nwrote {len(df)} rows across {df['clip_id'].nunique()} clips to {OUT_PATH}")


if __name__ == "__main__":
    main()
