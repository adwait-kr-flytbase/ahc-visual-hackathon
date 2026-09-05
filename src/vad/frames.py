"""Timestamp-based frame sampling. Train clip fps ranges 1.875 -> 30, so never index frames."""
import cv2
from PIL import Image


def probe(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    dur = (n / fps) if fps > 0 else 0.0
    cap.release()
    return {"fps": fps, "n_frames": int(n), "duration": dur}


def sample(path, t0, t1, n=16, max_side=640):
    """Return (frames, times) with n frames evenly spaced across [t0, t1)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], []
    span = max(t1 - t0, 1e-3)
    targets = [t0 + span * (i + 0.5) / n for i in range(n)]
    frames, times = [], []
    for t in targets:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        if not ok or img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        times.append(t)
    cap.release()
    if not frames:  # seeking failed; fall back to a sequential sweep
        frames, times = _sequential(path, t0, t1, n, max_side)
    return frames, times


def _sequential(path, t0, t1, n, max_side):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    span = max(t1 - t0, 1e-3)
    want = [int(round((t0 + span * (i + 0.5) / n) * fps)) for i in range(n)]
    want_set, frames, times, i = set(want), [], [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in want_set:
            h, w = img.shape[:2]
            if max(h, w) > max_side:
                s = max_side / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            times.append(i / fps)
        i += 1
        if len(frames) >= n:
            break
    cap.release()
    return frames, times
