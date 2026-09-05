"""Build ms-swift SFT rows from the short train clips and/or synthetic long videos.

    # free: the 3,173 short clips, already one-window-sized, no re-encoding
    python scripts/make_sft.py --clips --out out/sft

    # localisation: cut synthetic long videos into windows and label each
    python scripts/make_sft.py --synth out/synth-train --out out/sft --workers 8

Rows are built with `vad.prompts.build_sft_sample` -- never hand-rolled. If the training
template drifts from the inference template by one token the fine-tuned model scores worse
than zero-shot, so this is the single integration point that must not fork.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ahc_vad.groundtruth import load_ground_truth, load_manifest
from ahc_vad.synth import _load_real_durations
from vad.prompts import build_sft_sample
from vad.windows import plan

ROOT = Path(__file__).resolve().parents[1]


def to_container_path(path: Path | str, prefix: str) -> str:
    """Rewrite a local path to where the file will live inside the Modal container.

    Everything lands under the volume mount, and `out/` is stripped because the volume
    root holds `synth-train/`, `sft/` etc. directly rather than an `out/` level:
        dataset/train/fire/videos/X.mp4      -> /vol/dataset/train/fire/videos/X.mp4
        out/synth-train/videos/ST0001.mp4    -> /vol/synth-train/videos/ST0001.mp4
        out/sft/windows/ST0001/w0000.mp4     -> /vol/sft/windows/ST0001/w0000.mp4
    An empty prefix leaves paths repo-relative, for running locally.
    """
    relative = Path(path).resolve().relative_to(ROOT)
    parts = relative.parts
    if parts and parts[0] == "out":
        parts = parts[1:]
    joined = "/".join(parts)
    return f"{prefix.rstrip('/')}/{joined}" if prefix else joined


def clip_events_into(window: tuple[float, float], events, min_overlap: float = 0.5) -> list[dict]:
    """Re-clip absolute-time events into window-RELATIVE times, dropping slivers."""
    start, end = window
    out = []
    for event in events:
        if not event.is_localised:
            out.append({
                "class_name": event.class_name, "start": 0.0,
                "end": round(end - start, 2), "explanation": event.explanation or "",
            })
            continue
        lo = max(event.start_time_sec, start)
        hi = min(event.end_time_sec, end)
        if hi - lo >= min_overlap:
            out.append({
                "class_name": event.class_name,
                "start": round(lo - start, 2),
                "end": round(hi - start, 2),
                "explanation": event.explanation or "",
            })
    return out


def rows_from_clips(dataset: Path, prefix: str) -> list[dict]:
    """One row per short training clip. The clip IS the window -- nothing to re-encode."""
    real = _load_real_durations(ROOT / ".context" / "artifacts" / "videometa.tsv")
    rows = []
    for class_dir in sorted((dataset / "train").iterdir()):
        gt_path = class_dir / "ground_truth.csv"
        if not class_dir.is_dir() or not gt_path.exists():
            continue
        events_by_video = load_ground_truth(gt_path)
        for video_id, events in events_by_video.items():
            path = class_dir / "videos" / f"{video_id}.mp4"
            duration = real.get(video_id)
            if not path.exists() or duration is None:
                continue
            window = (0.0, duration)
            rows.append(build_sft_sample(
                to_container_path(path, prefix), clip_events_into(window, events), duration
            ))
    return rows


def _cut(job):
    source, start, end, out_path = job
    out_path = Path(out_path)
    if out_path.exists():
        return str(out_path), True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start}", "-t", f"{end - start}", "-i", str(source),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-an", str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return str(out_path), False
    return str(out_path), result.returncode == 0 and out_path.exists()


def rows_from_synth(synth_dir: Path, out_dir: Path, win: float, hop: float,
                    workers: int, prefix: str) -> list[dict]:
    """Cut each synthetic long video into windows and label each one."""
    gt = load_ground_truth(synth_dir / "ground_truth.csv")
    manifest = load_manifest(synth_dir / "manifest.json")
    window_dir = out_dir / "windows"

    jobs, meta = [], []
    for video_id, info in manifest.items():
        source = synth_dir / "videos" / f"{video_id}.mp4"
        if not source.exists():
            continue
        for index, window in enumerate(plan(info.duration_sec, win=win, hop=hop)):
            path = window_dir / video_id / f"w{index:04d}.mp4"
            jobs.append((str(source), window[0], window[1], str(path)))
            meta.append((video_id, window, path))

    print(f"  cutting {len(jobs)} windows with {workers} workers ...")
    ok_paths = set()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for done, (path, ok) in enumerate(executor.map(_cut, jobs), 1):
            if ok:
                ok_paths.add(path)
            if done % 250 == 0:
                print(f"    {done}/{len(jobs)}")

    rows = []
    for video_id, window, path in meta:
        if str(path) not in ok_paths:
            continue
        events = clip_events_into(window, gt.get(video_id, []))
        rows.append(build_sft_sample(
            to_container_path(path, prefix), events, window[1] - window[0]
        ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", action="store_true", help="include the short train clips")
    parser.add_argument("--synth", type=Path, action="append", default=[],
                        help="a synthetic dataset dir; repeatable")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--win", type=float, default=20.0)
    parser.add_argument("--hop", type=float, default=20.0,
                        help="20 = non-overlapping; inference uses 10 but training needs fewer rows")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--path-prefix", default="/vol",
                        help="container mount the video paths resolve under; \"\" for repo-relative")
    args = parser.parse_args()

    rows: list[dict] = []
    if args.clips:
        print("building rows from short train clips ...")
        clip_rows = rows_from_clips(args.dataset, args.path_prefix)
        print(f"  {len(clip_rows)} rows")
        rows += clip_rows
    for synth_dir in args.synth:
        print(f"building rows from {synth_dir} ...")
        synth_rows = rows_from_synth(synth_dir, args.out, args.win, args.hop,
                                     args.workers, args.path_prefix)
        print(f"  {len(synth_rows)} rows")
        rows += synth_rows

    if not rows:
        print("nothing to do: pass --clips and/or --synth", file=sys.stderr)
        return 1

    random.Random(args.seed).shuffle(rows)
    cut = max(1, int(len(rows) * args.val_fraction))
    val, train = rows[:cut], rows[cut:]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("val", val)):
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        empties = sum(1 for r in subset if '"events": []' in r["messages"][2]["content"])
        print(f"wrote {path}: {len(subset)} rows, {empties} negatives ({100*empties/len(subset):.0f}%)")
    sample = (train or val)[0]["videos"][0]
    print(f"video paths resolve as: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
