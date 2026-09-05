"""Rebuild ground_truth.csv / videos.csv / manifest.json for already-rendered videos.

Compositions are deterministic in (seed, index), so a run killed before it wrote its
metadata can be recovered without re-rendering. Each video is ffprobed and dropped if its
real duration disagrees with the composition by more than `--tolerance` -- that catches the
file the process was midway through writing when it died.

    python scripts/recover_synth_metadata.py --out out/synth-train --split train --n 200
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ahc_vad.synth import load_clip_pool, write_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_synthetic import plan_one, split_pool  # noqa: E402


def real_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        ).stdout
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--tolerance", type=float, default=2.0)
    args = parser.parse_args()

    pool = load_clip_pool(args.dataset)
    train_pool, dev_pool = split_pool(pool, args.dev_fraction, args.seed)
    chosen = dev_pool if args.split == "dev" else train_pool
    prefix = "SD" if args.split == "dev" else "ST"

    good, durations, dropped = [], {}, []
    for index in range(args.n):
        composition, _ = plan_one(index, prefix, chosen, args.seed)
        path = args.out / "videos" / f"{composition.video_id}.mp4"
        if not path.exists():
            continue
        actual = real_duration(path)
        if actual is None or abs(actual - composition.duration) > args.tolerance:
            dropped.append((composition.video_id, composition.duration, actual))
            path.unlink(missing_ok=True)
            continue
        good.append(composition)
        durations[composition.video_id] = actual

    write_dataset(good, args.out, durations)
    events = sum(len(c.events()) for c in good)
    print(f"recovered {len(good)} videos, {events} events -> {args.out}")
    if dropped:
        print(f"dropped {len(dropped)} (duration mismatch = killed mid-write):")
        for vid, declared, actual in dropped[:5]:
            print(f"  {vid}: composed {declared:.1f}s, on disk {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
