"""Permutation null for the spike proposals.

"35% of our spikes land near an event start" means nothing until you know what a random spike
would score. This shuffles the proposal times within each video, keeping their number per video
fixed, and reports where the real result falls in that null distribution.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ahc_vad.groundtruth import load_ground_truth, load_manifest  # noqa: E402

WINDOW = 15.0
TRIALS = 2000
rng = np.random.default_rng(0)

manifest = load_manifest(ROOT / "data/manifest.json")
gt = load_ground_truth(ROOT / "dataset/test/ground_truth.csv")
videos = {}
for line in (ROOT / "mv/out/features.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        if not r.get("failed"):
            videos[r["video_id"]] = r

def robust_z(v):
    v = np.asarray(v, float); m = np.median(v)
    mad = np.median(np.abs(v - m))
    return (v - m) / (mad * 1.4826 if mad > 1e-9 else (v.std() or 1.0))

def group(times):
    out = []
    for t in times:
        if not out or t - out[-1][-1] > 2: out.append([t])
        else: out[-1].append(t)
    return [g[0] for g in out]

print(f"Permutation null, {TRIALS} trials, +/-{WINDOW:.0f}s tolerance\n")
print(f"  {'z':>5}{'proposals':>11}{'real P':>9}{'null P':>9}{'p-val':>8}"
      f"{'real R':>9}{'null R':>9}{'p-val':>8}")

for threshold in (2.0, 2.5, 3.0):
    per_video = []
    for vid in sorted(videos):
        if manifest[vid].level == 1: continue
        series = videos[vid]["series"]
        starts = [e.start_time_sec for e in gt.get(vid, []) if e.is_localised]
        if not series: continue
        times = np.array([s["t"] for s in series], float)
        props = group(times[robust_z([s["residual"] for s in series]) > threshold])
        per_video.append((times, starts, len(props), props))

    def tally(picker):
        hit = near = spikes = starts_n = 0
        for times, starts, n, real in per_video:
            props = picker(times, n, real)
            spikes += len(props); starts_n += len(starts)
            hit += sum(any(abs(p - s) <= WINDOW for p in props) for s in starts)
            near += sum(any(abs(p - s) <= WINDOW for s in starts) for p in props)
        return (near / spikes if spikes else 0.0), (hit / starts_n if starts_n else 0.0)

    real_p, real_r = tally(lambda t, n, real: real)
    null_p, null_r = [], []
    for _ in range(TRIALS):
        p, r = tally(lambda t, n, real: rng.choice(t, size=min(n, len(t)), replace=False))
        null_p.append(p); null_r.append(r)
    null_p, null_r = np.array(null_p), np.array(null_r)
    # one-sided: how often does chance match or beat us?
    pv_p = float((null_p >= real_p).mean()); pv_r = float((null_r >= real_r).mean())
    total = sum(n for _, _, n, _ in per_video)
    print(f"  {threshold:>5.1f}{total:>11}{real_p:>9.3f}{null_p.mean():>9.3f}{pv_p:>8.3f}"
          f"{real_r:>9.3f}{null_r.mean():>9.3f}{pv_r:>8.3f}")

print("\np-val is the share of random trials that did as well or better.")
print("Above ~0.05 means the real proposals are indistinguishable from randomly placed ones.")
