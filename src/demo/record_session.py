"""Record short labelled webcam clips for the self-collected validation set.

Labels here are DELIBERATE BEHAVIOURAL STATES, not engagement. Rounds 5-7
showed the geometric features track a movement-vs-stillness axis rather than
DAiSEE's engagement label, so this records the axis we can actually test.

Controls:
    1 / 2 / 3   choose the state to record
    space       record one clip of the chosen state
    q           quit
"""

import argparse
import os
import time
from datetime import datetime

import cv2

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_ROOT = os.path.join(REPO_ROOT, "data", "self_collected")

STATES = {
    ord("1"): ("still_attentive", "sit still, eyes on screen"),
    ord("2"): ("fidgety_distracted", "move around, look away, shift posture"),
    ord("3"): ("drowsy", "slow blinks, head dropping, eyes closing"),
}

CLIP_SECONDS = 10
TARGET_FPS = 30


class CameraSource:
    """Thin wrapper so the record loop can be driven by a fake source in tests."""

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open camera {index}. on macOS check that the terminal "
                "has camera permission in system settings > privacy & security"
            )

    def read(self):
        return self.cap.read()

    def fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1 else TARGET_FPS

    def frame_size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self):
        self.cap.release()


def clip_path(state):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, state)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{state}_{stamp}.mp4")


def count_existing(state):
    out_dir = os.path.join(OUT_ROOT, state)
    if not os.path.isdir(out_dir):
        return 0
    return len([f for f in os.listdir(out_dir) if f.lower().endswith((".mp4", ".avi"))])


def record_clip(source, state, seconds=CLIP_SECONDS, show=True):
    """Record one clip. Returns the written path, or None if the capture failed."""
    fps = source.fps()
    width, height = source.frame_size()
    path = clip_path(state)

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")

    n_frames = int(seconds * fps)
    started = time.time()
    written = 0

    for i in range(n_frames):
        ok, frame = source.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

        if show:
            preview = frame.copy()
            remaining = seconds - (time.time() - started)
            cv2.putText(preview, f"REC {state}  {remaining:4.1f}s", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("neuroright recorder", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    writer.release()

    if written == 0:
        os.remove(path)
        return None
    return path


def run(source, show=True, seconds=CLIP_SECONDS):
    state = "still_attentive"
    print(__doc__)
    for key, (name, hint) in STATES.items():
        print(f"  {chr(key)} = {name:20s} ({hint})   {count_existing(name)} clips recorded")

    while True:
        ok, frame = source.read()
        if not ok:
            print("camera stopped returning frames")
            break

        if show:
            preview = frame.copy()
            cv2.putText(preview, f"state: {state}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.putText(preview, "1/2/3 pick state   space record   q quit", (12, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("neuroright recorder", preview)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = ord("q")

        if key == ord("q"):
            break
        if key in STATES:
            state = STATES[key][0]
            print(f"state -> {state}")
        elif key == ord(" "):
            path = record_clip(source, state, seconds=seconds, show=show)
            if path:
                print(f"saved {path}  ({count_existing(state)} clips of {state})")
            else:
                print("capture failed, clip discarded")

    source.release()
    if show:
        cv2.destroyAllWindows()

    print("\nrecorded so far:")
    for _, (name, _) in STATES.items():
        print(f"  {name:20s} {count_existing(name)} clips")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=CLIP_SECONDS)
    args = parser.parse_args()

    run(CameraSource(args.camera), seconds=args.seconds)


if __name__ == "__main__":
    main()
