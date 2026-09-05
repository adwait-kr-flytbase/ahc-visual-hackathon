"""Cross-model consensus over independent runs.

The portal's own diagnostic says precision 14% with 67 false alarms and that cutting
false alarms pays more than finding events. Union raises recall and makes that worse.
Consensus is the opposite lever and costs no GPU: keep an event only when >= k
independent runs agree on the class at overlapping times.
"""
from .prompts import CLASSES


def _overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def consensus(runs, duration, min_agree=2, iou_join=0.1):
    """runs: list of event-lists from independent runs of the same video."""
    pool = [dict(e, _run=i) for i, evs in enumerate(runs) for e in evs]
    out = []
    for cls in CLASSES:
        xs = sorted([e for e in pool if e["class_name"] == cls], key=lambda e: e["start"])
        used = [False] * len(xs)
        for i, seed in enumerate(xs):
            if used[i]:
                continue
            group, runs_seen = [seed], {seed["_run"]}
            used[i] = True
            for j in range(i + 1, len(xs)):
                if used[j]:
                    continue
                ov = _overlap(seed, xs[j])
                span = max(seed["end"] - seed["start"], xs[j]["end"] - xs[j]["start"], 1e-6)
                if ov / span >= iou_join:
                    group.append(xs[j]); runs_seen.add(xs[j]["_run"]); used[j] = True
            if len(runs_seen) >= min_agree:
                out.append({
                    "class_name": cls,
                    "start": round(min(g["start"] for g in group), 2),
                    "end": round(min(duration, max(g["end"] for g in group)), 2),
                    "confidence": round(max(g["confidence"] for g in group), 3),
                    "explanation": max(group, key=lambda g: g["confidence"]).get("explanation", ""),
                    "n_agree": len(runs_seen),
                })
    out.sort(key=lambda e: e["start"])
    return out
