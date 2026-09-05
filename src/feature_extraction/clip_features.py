"""Clip-level temporal features built on top of the per-frame face features.

Round 5 showed that per-frame means alone plateau, so this adds the dynamics
the static aggregates were missing: how often the subject blinks, how much the
gaze moves, and how fast the head turns.
"""

import numpy as np
import pandas as pd

FRAME_FEATURES = [
    "ear_left", "ear_right", "ear_mean",
    "gaze_left_x", "gaze_right_x", "gaze_x_mean",
    "gaze_left_y", "gaze_right_y", "gaze_y_mean",
    "head_pitch", "head_yaw", "head_roll",
]

POSE_COLS = ["head_pitch", "head_yaw", "head_roll"]
GAZE_COLS = ["gaze_x_mean", "gaze_y_mean"]

SAMPLE_FPS = 5
# EAR below this counts as a closed eye; DAiSEE faces sit around 0.20-0.25 open
BLINK_EAR_THRESHOLD = 0.15


def blink_stats(ear_series, fps=SAMPLE_FPS, threshold=BLINK_EAR_THRESHOLD):
    """Count threshold crossings (open -> closed transitions) as blinks.

    At 5fps this undercounts genuinely fast blinks, so treat it as a
    'eye-closure rate' proxy rather than a clinically exact blink count.
    """
    ear = np.asarray(ear_series, dtype=float)
    closed = ear < threshold
    # a blink is a transition from open to closed
    onsets = int(np.sum(~closed[:-1] & closed[1:])) if len(closed) > 1 else 0
    duration_sec = len(ear) / fps if len(ear) else 0.0
    return {
        "blink_count": onsets,
        "blink_rate_hz": onsets / duration_sec if duration_sec > 0 else 0.0,
        "closed_frame_fraction": float(np.mean(closed)) if len(closed) else 0.0,
        "ear_min": float(np.min(ear)) if len(ear) else 0.0,
        "ear_range": float(np.ptp(ear)) if len(ear) else 0.0,
    }


def velocity_stats(values, prefix):
    """Frame-to-frame absolute change: how fast this signal moves."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return {f"{prefix}_vel_mean": 0.0, f"{prefix}_vel_std": 0.0, f"{prefix}_vel_max": 0.0}
    diffs = np.abs(np.diff(arr))
    return {
        f"{prefix}_vel_mean": float(np.mean(diffs)),
        f"{prefix}_vel_std": float(np.std(diffs)),
        f"{prefix}_vel_max": float(np.max(diffs)),
    }


def clip_features(group):
    """Enriched feature vector for one clip's frame sequence."""
    feats = {}

    # keep the round 5 aggregates: this is additive
    for col in FRAME_FEATURES:
        values = group[col].to_numpy(dtype=float)
        feats[f"{col}_mean"] = float(np.mean(values))
        feats[f"{col}_std"] = float(np.std(values))
        feats[f"{col}_min"] = float(np.min(values))
        feats[f"{col}_max"] = float(np.max(values))

    feats.update(blink_stats(group["ear_mean"]))

    # gaze dispersion, horizontal and vertical
    for col in GAZE_COLS:
        values = group[col].to_numpy(dtype=float)
        feats[f"{col}_var"] = float(np.var(values))
        feats.update(velocity_stats(values, col))
    gx = group["gaze_x_mean"].to_numpy(dtype=float)
    gy = group["gaze_y_mean"].to_numpy(dtype=float)
    feats["gaze_dispersion"] = float(np.sqrt(np.var(gx) + np.var(gy)))

    # head-pose dynamics
    for col in POSE_COLS:
        values = group[col].to_numpy(dtype=float)
        feats[f"{col}_var"] = float(np.var(values))
        feats.update(velocity_stats(values, col))
    pose_vel = np.stack([
        np.abs(np.diff(group[c].to_numpy(dtype=float))) for c in POSE_COLS
    ]) if len(group) > 1 else np.zeros((3, 1))
    feats["head_motion_magnitude"] = float(np.mean(np.linalg.norm(pose_vel, axis=0)))

    return feats


def build_clip_table(df, label_cols=("boredom", "engagement", "confusion", "frustration")):
    """Turn a per-frame sequence table into one enriched row per clip."""
    rows = []
    for clip_id, group in df.sort_values(["clip_id", "frame_idx"]).groupby("clip_id"):
        row = {"clip_id": clip_id}
        row.update(clip_features(group))
        for col in label_cols:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("clip_id")


def feature_columns(table, label_cols=("boredom", "engagement", "confusion", "frustration")):
    return [c for c in table.columns if c not in label_cols]
