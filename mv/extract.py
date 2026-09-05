"""Ego-motion-robust motion features from H.264 motion vectors. CPU only, no GPU, no model.

Every anomaly-detection paper in the reading list gates on frame differencing, which assumes a
static background. A third of this dataset is drone and a third is dashcam, so that gate fires on
the whole frame the moment the camera moves.

Motion vectors come free out of the decoder -- the encoder already computed them. Fit a global
affine to the MV field with RANSAC: the fit is the camera's own motion, and the blocks RANSAC
rejects are things moving independently of the camera. That residual is the signal.

    python mv/extract.py --videos T025 T026 --out mv/out/features.jsonl

Writes one JSON object per video, flushed as each finishes, so a long run is readable early.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def frame_features(mv, width, height):
    """One frame of motion vectors -> (global camera motion, residual stats).

    RANSAC fits the dominant motion, which for a moving camera is the background. Blocks that
    do not fit are moving independently of it. On a static camera the fit degenerates to zero
    translation and the outliers are simply the moving objects, so the same code covers both.
    """
    # source < 0 is prediction from a past reference. B-frames also carry future-referenced
    # vectors whose sign is inverted; mixing them would cancel real motion out.
    mv = mv[mv["source"] < 0]
    if len(mv) < 24:
        return None

    scale = mv["motion_scale"].astype(np.float64)
    dx = mv["motion_x"] / scale
    dy = mv["motion_y"] / scale
    src = np.column_stack([mv["dst_x"].astype(np.float64) - dx,
                           mv["dst_y"].astype(np.float64) - dy])
    dst = np.column_stack([mv["dst_x"].astype(np.float64),
                           mv["dst_y"].astype(np.float64)])

    matrix, inliers = cv2.estimateAffine2D(
        src.reshape(-1, 1, 2).astype(np.float32),
        dst.reshape(-1, 1, 2).astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=400, confidence=0.985,
    )
    if matrix is None:
        return None
    inliers = inliers.ravel().astype(bool)

    predicted = src @ matrix[:, :2].T + matrix[:, 2]
    residual = np.linalg.norm(dst - predicted, axis=1)
    diagonal = float(np.hypot(width, height))

    # The camera's own motion, read off the fit at the frame centre.
    centre = np.array([[width / 2, height / 2]], dtype=np.float64)
    ego = float(np.linalg.norm(centre @ matrix[:, :2].T + matrix[:, 2] - centre)) / diagonal

    outlier = ~inliers
    count = int(outlier.sum())
    stats = {
        "blocks": int(len(mv)),
        "ego": ego,
        # Residual energy carried by independently-moving blocks, as a share of the frame.
        "residual": float(residual[outlier].sum()) / diagonal / max(len(mv), 1) * 1000,
        "outlier_frac": count / len(mv),
        "spread": 0.0,
        "blobs": 0,
    }
    if count >= 3:
        points = dst[outlier]
        stats["spread"] = float(np.hypot(points[:, 0].std(), points[:, 1].std())) / diagonal
        stats["blobs"] = count_blobs(points, width, height)
    return stats


def count_blobs(points, width, height, cell=32):
    """Independently-moving blocks that touch each other are one object, not many."""
    grid = np.zeros((int(height // cell) + 2, int(width // cell) + 2), np.uint8)
    ys = np.clip((points[:, 1] // cell).astype(int), 0, grid.shape[0] - 1)
    xs = np.clip((points[:, 0] // cell).astype(int), 0, grid.shape[1] - 1)
    grid[ys, xs] = 1
    n, _ = cv2.connectedComponents(grid, connectivity=8)
    return int(n - 1)


def video_features(path, max_seconds=None, stride=1):
    """Per-second aggregates of the residual motion field.

    `stride` analyses every Nth motion-vector frame. Every frame still has to be decoded to get
    its vectors, but the RANSAC fit dominates the cost, and a few frames a second is plenty for
    a per-second series.
    """
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}
    stream.thread_type = "AUTO"
    width, height = stream.codec_context.width, stream.codec_context.height
    time_base = float(stream.time_base)

    buckets = defaultdict(list)
    frames = decoded = seen = 0
    for frame in container.decode(stream):
        decoded += 1
        side = frame.side_data.get("MOTION_VECTORS")
        if side is None:
            continue
        seen += 1
        if seen % stride:
            continue
        second = int(frame.pts * time_base) if frame.pts is not None else None
        if second is None:
            continue
        if max_seconds and second > max_seconds:
            break
        stats = frame_features(side.to_ndarray(), width, height)
        if stats:
            buckets[second].append(stats)
            frames += 1
    container.close()

    series = []
    for second in sorted(buckets):
        rows = buckets[second]
        series.append({
            "t": second,
            "residual": float(np.mean([r["residual"] for r in rows])),
            "outlier_frac": float(np.mean([r["outlier_frac"] for r in rows])),
            "spread": float(np.mean([r["spread"] for r in rows])),
            "blobs": float(np.mean([r["blobs"] for r in rows])),
            "ego": float(np.mean([r["ego"] for r in rows])),
            "n": len(rows),
        })
    return {"width": width, "height": height, "frames_with_mvs": frames,
            "frames_decoded": decoded, "series": series}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="*", help="video ids; default every id in the manifest")
    parser.add_argument("--dir", type=Path, default=ROOT / "dataset/test/videos")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "mv/out/features.jsonl")
    parser.add_argument("--max-seconds", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1, help="analyse every Nth MV frame")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    manifest = {v["video_id"]: v for v in json.loads(args.manifest.read_text())["videos"]}
    ids = args.videos or list(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(vid, args.dir / f"{vid}.mp4", manifest[vid], args.max_seconds, args.stride)
            for vid in ids]
    missing = [vid for vid, path, *_ in jobs if not path.exists()]
    for vid in missing:
        print(f"MISSING {vid}", flush=True)
    jobs = [j for j in jobs if j[1].exists()]

    done = 0
    with args.out.open("w") as handle:
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                results = pool.map(_one, jobs)
                for result in results:
                    done += 1
                    _emit(handle, result, done, len(jobs))
        else:
            for job in jobs:
                done += 1
                _emit(handle, _one(job), done, len(jobs))
    return 0


def _one(job):
    """Extract one video. Returns a result dict; never raises, so one bad file cannot lose the rest."""
    video_id, path, info, max_seconds, stride = job
    started = time.time()
    try:
        result = video_features(path, max_seconds, stride)
    except Exception as exc:
        return {"video_id": video_id, "failed": f"{type(exc).__name__}: {exc}"}
    result.update(video_id=video_id, level=info["level"], duration=info["duration_sec"],
                  seconds_to_extract=time.time() - started)
    return result


def _emit(handle, result, done, total):
    handle.write(json.dumps(result) + "\n")
    handle.flush()
    if result.get("failed"):
        print(f"[{done}/{total}] {result['video_id']} FAILED {result['failed']}", flush=True)
        return
    series = result["series"]
    ego = float(np.mean([s["ego"] for s in series])) if series else 0.0
    res = float(np.mean([s["residual"] for s in series])) if series else 0.0
    print(f"[{done}/{total}] {result['video_id']} L{result['level']} {len(series)}s  "
          f"{result['frames_with_mvs']}/{result['frames_decoded']} frames  "
          f"ego={ego:.4f} residual={res:.3f}  {result['seconds_to_extract']:.1f}s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
