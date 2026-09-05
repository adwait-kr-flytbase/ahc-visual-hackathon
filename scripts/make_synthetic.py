"""Generate synthetic long multi-event videos with exact ground truth.

    # smoke test: 2 videos
    python scripts/make_synthetic.py --n 2 --out out/synth-smoke --split dev

    # the held-out dev set the portal cannot score
    python scripts/make_synthetic.py --n 100 --out out/synth-dev --split dev --workers 8

Source clips are partitioned into disjoint `train` and `dev` pools BEFORE composition, so
a dev video never reuses a clip seen in training.
"""

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ahc_vad.synth import TEST_PROFILES, compose, load_clip_pool, render, write_dataset
from ahc_vad.taxonomy import ANOMALY_CLASSES

ROOT = Path(__file__).resolve().parents[1]

# D2 in the public pack is always 240 s; D3 spans 307-629 s.
LEVEL_DURATIONS = {2: (240.0, 240.0), 3: (300.0, 630.0)}


def split_pool(pool: dict, dev_fraction: float, seed: int) -> tuple[dict, dict]:
    """Partition each class's clips into disjoint train / dev pools."""
    train_pool, dev_pool = {}, {}
    for name, clips in pool.items():
        shuffled = list(clips)
        random.Random(seed + hash(name) % 10_000).shuffle(shuffled)
        cut = max(1, int(len(shuffled) * dev_fraction))
        dev_pool[name] = shuffled[:cut]
        train_pool[name] = shuffled[cut:] or shuffled[:cut]
    return train_pool, dev_pool


def plan_one(index: int, prefix: str, pool: dict, seed: int):
    rng = random.Random(seed * 1_000_003 + index)
    level = 2 if rng.random() < 0.6 else 3
    low, high = LEVEL_DURATIONS[level]
    target = rng.uniform(low, high)

    # ~12% of videos carry no events at all -- the false-alarm negatives.
    if rng.random() < 0.12:
        names = []
    else:
        available = sorted(set(pool) & ANOMALY_CLASSES)
        count = rng.choice([1, 1, 2, 2, 3, 4, 6])
        if rng.random() < 0.45:  # single-class runs, the T025/T027/T028 pattern
            names = [rng.choice(available)] * count
        else:  # mixed classes in one video, the T026 pattern
            names = [rng.choice(available) for _ in range(count)]

    video_id = f"{prefix}{index:04d}"
    profile = rng.choices(TEST_PROFILES, weights=[p[2] for p in TEST_PROFILES])[0]
    return compose(video_id, level, target, names, pool, rng), profile


def _render_job(args):
    composition, profile, out_dir, crf = args
    path = Path(out_dir) / "videos" / f"{composition.video_id}.mp4"
    ok = render(composition, path, profile, crf=crf)
    return composition.video_id, ok, composition.duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--crf", type=int, default=30)
    args = parser.parse_args()

    print(f"loading clip pool from {args.dataset}/train ...")
    pool = load_clip_pool(args.dataset)
    print(f"  {sum(len(v) for v in pool.values())} clips across {len(pool)} classes")

    train_pool, dev_pool = split_pool(pool, args.dev_fraction, args.seed)
    chosen = dev_pool if args.split == "dev" else train_pool
    print(f"  using the {args.split} pool: {sum(len(v) for v in chosen.values())} clips")

    prefix = "SD" if args.split == "dev" else "ST"
    planned = [plan_one(i, prefix, chosen, args.seed) for i in range(args.n)]

    jobs = [(c, p, str(args.out), args.crf) for c, p in planned]
    durations, failures = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for done, (video_id, ok, duration) in enumerate(executor.map(_render_job, jobs), 1):
            if ok:
                durations[video_id] = duration
            else:
                failures.append(video_id)
            if done % 10 == 0 or done == len(jobs):
                print(f"  rendered {done}/{len(jobs)}")

    good = [c for c, _ in planned if c.video_id in durations]
    write_dataset(good, args.out, durations)

    events = sum(len(c.events()) for c in good)
    total = sum(c.duration for c in good)
    print(f"\nwrote {len(good)} videos to {args.out}")
    print(f"  {events} events, {total / 60:.1f} min of video")
    print(f"  normal (no-event) videos: {sum(1 for c in good if not c.events())}")
    if failures:
        print(f"  FAILED: {len(failures)} -> {failures[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
