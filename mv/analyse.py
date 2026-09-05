"""Does residual motion energy actually mark anomaly events? Measured, not assumed.

Reads mv/out/features.jsonl and dataset/test/ground_truth.csv and answers three questions:

  1. Within a video, is a second that falls inside a ground-truth event ranked higher by residual
     motion than a second outside one? Reported as AUC, which needs no threshold. 0.5 is chance.
  2. If we call the top-scoring seconds "spikes", what fraction of event starts do they catch, and
     what fraction of spikes land near an event start?
  3. Does any of it survive on the long D2/D3 videos, where 75 of the 100 marks are?

Only D2/D3 carry timestamps, so only they can answer this -- D1 ground truth has no timing.

    python mv/analyse.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ahc_vad.groundtruth import load_ground_truth, load_manifest  # noqa: E402

FEATURES = ["residual", "outlier_frac", "spread", "blobs"]


def robust_z(values):
    """Median/MAD standardisation. Robust because a long event would drag a mean-based z toward
    itself and hide exactly the thing we are looking for."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = mad * 1.4826 if mad > 1e-9 else (values.std() or 1.0)
    return (values - median) / scale


def auc(scores, labels):
    """Rank-based AUC. Chance is 0.5; None when one class is absent."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return None
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties, or repeated identical scores inflate the result
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "mv/out/features.jsonl")
    parser.add_argument("--gt", type=Path, default=ROOT / "dataset/test/ground_truth.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    parser.add_argument("--window", type=float, default=15.0,
                        help="seconds either side of an event start that a spike may claim; "
                             "15 matches the portal's D2 boundary tolerance")
    parser.add_argument("--out", type=Path, default=ROOT / "mv/out/report.json")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    ground_truth = load_ground_truth(args.gt)
    videos = {}
    for line in args.features.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if not row.get("failed"):
                videos[row["video_id"]] = row

    timed = [v for v in sorted(videos) if manifest[v].level > 1]
    print(f"{len(videos)} videos extracted, {len(timed)} of them timed (D2/D3)\n")

    # ---- ego-motion, which the manifest leaves blank for every video ----
    print("Ego-motion per video (the `domain` field the manifest ships empty)")
    egos = []
    for vid in sorted(videos):
        series = videos[vid]["series"]
        if not series:
            continue
        ego = float(np.median([s["ego"] for s in series]))
        egos.append((vid, manifest[vid].level, ego))
    moving = [e for e in egos if e[2] > 0.0005]
    print(f"  {len(moving)} of {len(egos)} videos have a moving camera (median ego > 5e-4)")
    print("  most static :", ", ".join(f"{v}={e:.5f}" for v, _, e in sorted(egos, key=lambda x: x[2])[:5]))
    print("  most moving :", ", ".join(f"{v}={e:.5f}" for v, _, e in sorted(egos, key=lambda x: -x[2])[:5]))

    # ---- Q1: does residual rank in-event seconds above out-of-event ones? ----
    print(f"\nAUC per video -- can the feature separate in-event seconds from the rest?")
    print(f"  {'video':7}{'lvl':4}{'secs':6}{'in-evt':7}" + "".join(f"{f:>14}" for f in FEATURES))
    per_feature = {f: [] for f in FEATURES}
    rows = []
    for vid in timed:
        series = videos[vid]["series"]
        events = [(e.start_time_sec, e.end_time_sec) for e in ground_truth.get(vid, [])
                  if e.is_localised]
        if not series:
            continue
        times = np.array([s["t"] for s in series], float)
        labels = np.zeros(len(times), bool)
        for start, end in events:
            labels |= (times >= start) & (times <= end)
        line = f"  {vid:7}{manifest[vid].level:<4}{len(times):<6}{int(labels.sum()):<7}"
        entry = {"video_id": vid, "level": manifest[vid].level, "seconds": len(times),
                 "in_event_seconds": int(labels.sum()), "auc": {}}
        for feature in FEATURES:
            value = auc([s[feature] for s in series], labels)
            entry["auc"][feature] = value
            line += f"{'  --  ' if value is None else f'{value:.3f}':>14}"
            if value is not None:
                per_feature[feature].append(value)
        rows.append(entry)
        print(line)

    print(f"\n  {'MEAN':7}{'':4}{'':6}{'':7}" + "".join(
        f"{(f'{np.mean(per_feature[f]):.3f}' if per_feature[f] else '--'):>14}" for f in FEATURES))
    print(f"  {'n':7}{'':4}{'':6}{'':7}" + "".join(f"{len(per_feature[f]):>14}" for f in FEATURES))

    # ---- Q2: spikes vs event starts ----
    print(f"\nSpike analysis, +/-{args.window:.0f}s tolerance, spike = robust z above threshold")
    print(f"  {'z':>5}{'spikes':>9}{'starts':>8}{'caught':>8}{'P(start|spike)':>16}{'P(spike|start)':>16}")
    spike_rows = []
    for threshold in (2.0, 2.5, 3.0, 4.0):
        total_spikes = caught = total_starts = spikes_near = 0
        for vid in timed:
            series = videos[vid]["series"]
            starts = [e.start_time_sec for e in ground_truth.get(vid, []) if e.is_localised]
            if not series:
                continue
            times = np.array([s["t"] for s in series], float)
            z = robust_z([s["residual"] for s in series])
            spike_times = times[z > threshold]
            # collapse runs of adjacent spike seconds into one proposal
            grouped = []
            for t in spike_times:
                if not grouped or t - grouped[-1][-1] > 2:
                    grouped.append([t])
                else:
                    grouped[-1].append(t)
            proposals = [g[0] for g in grouped]
            total_spikes += len(proposals)
            total_starts += len(starts)
            caught += sum(any(abs(p - s) <= args.window for p in proposals) for s in starts)
            spikes_near += sum(any(abs(p - s) <= args.window for s in starts) for p in proposals)
        precision = spikes_near / total_spikes if total_spikes else None
        recall = caught / total_starts if total_starts else None
        spike_rows.append({"z": threshold, "spikes": total_spikes, "starts": total_starts,
                           "caught": caught, "precision": precision, "recall": recall})
        fmt = lambda x: "  --  " if x is None else f"{x:.3f}"
        print(f"  {threshold:>5.1f}{total_spikes:>9}{total_starts:>8}{caught:>8}"
              f"{fmt(precision):>16}{fmt(recall):>16}")

    baseline = sum(r["in_event_seconds"] for r in rows) / max(sum(r["seconds"] for r in rows), 1)
    print(f"\n  Base rate: {baseline:.3f} of all timed seconds fall inside an event.")
    print("  A spike placed at random would reach about that precision.")

    args.out.write_text(json.dumps(
        {"per_video": rows, "spikes": spike_rows,
         "mean_auc": {f: (float(np.mean(v)) if v else None) for f, v in per_feature.items()},
         "ego": [{"video_id": v, "level": l, "ego": e} for v, l, e in egos],
         "base_rate": baseline, "window_sec": args.window}, indent=2))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
