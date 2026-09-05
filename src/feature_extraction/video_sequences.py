"""Turn a video file into a per-frame feature sequence.

Shared by the DAiSEE extraction and the self-collected demo sessions so both
paths sample and featurise video identically.
"""

import cv2
import pandas as pd

from face_features import FaceFeatureExtractor

SAMPLE_FPS = 5


def sample_frames(video_path, sample_fps=SAMPLE_FPS):
    """Read a video and keep roughly `sample_fps` frames per second."""
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


def extract_sequence(video_path, extractor=None, sample_fps=SAMPLE_FPS):
    """Per-frame feature rows for one video. Frames with no detected face are
    dropped, so an empty frame list means the face was never found."""
    own_extractor = extractor is None
    if own_extractor:
        extractor = FaceFeatureExtractor()

    rows = []
    frames = sample_frames(video_path, sample_fps)
    for frame_idx, frame in frames:
        features = extractor.extract(frame)
        if features is None:
            continue
        rows.append({"frame_idx": frame_idx, **features})

    if own_extractor:
        extractor.close()

    return pd.DataFrame(rows), len(frames)
