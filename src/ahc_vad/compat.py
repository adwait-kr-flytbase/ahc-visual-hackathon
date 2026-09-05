"""Drop-in adapter for the `vad/` pipeline's scorer signature.

vad/score.py documents the contract:
    score(pred_jsonl_path, gt_csv_path, manifest_path) -> dict

This exposes the tested matcher behind that exact signature, reading the events.jsonl
that vad/run.py emits. To adopt it, replace the body of vad/score.py's `score` with:

    from ahc_vad.compat import score_events_jsonl as score

Output keys match vad/score.py's: {"levels", "classes", "overall"}.
"""

import json
from pathlib import Path

from ahc_vad.events import Event
from ahc_vad.groundtruth import load_ground_truth, load_manifest
from ahc_vad.scoring import DifficultyScore, MatchPolicy, score


def _events_from_jsonl(path: str | Path) -> dict[str, list[Event]]:
    """Read vad/run.py's events.jsonl. Rows use `start`/`end`, not `*_time_sec`."""
    predictions: dict[str, list[Event]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        predictions[row["video_id"]] = [
            Event(
                class_name=event["class_name"],
                start_time_sec=event.get("start", event.get("start_time_sec")),
                end_time_sec=event.get("end", event.get("end_time_sec")),
                explanation=event.get("explanation"),
            )
            for event in row.get("events", [])
        ]
    return predictions


def _pr(entry: DifficultyScore) -> dict:
    return {
        "precision": round(entry.precision, 4),
        "recall": round(entry.recall, 4),
        "f1": round(entry.f1, 4),
        "found": entry.true_positives,
        "fa": entry.false_alarms,
        "missed": entry.misses,
    }


def score_events_jsonl(
    pred_jsonl: str | Path,
    gt_csv: str | Path,
    manifest: str | Path,
    *,
    boundary_tolerance_sec: float | None = None,
) -> dict:
    """Match vad/score.py's signature and output shape, backed by the tested matcher."""
    report = score(
        load_ground_truth(gt_csv),
        _events_from_jsonl(pred_jsonl),
        load_manifest(manifest),
        policy_d23=MatchPolicy(boundary_tolerance_sec=boundary_tolerance_sec),
    )
    totals = [0, 0, 0]
    for entry in report.by_difficulty.values():
        totals[0] += entry.true_positives
        totals[1] += entry.false_alarms
        totals[2] += entry.misses
    return {
        "levels": {f"L{level}": _pr(e) for level, e in report.by_difficulty.items()},
        "classes": {name: _pr(e) for name, e in report.by_class.items()},
        "overall": _pr(DifficultyScore(0, *totals)),
        "proxy_score": round(report.proxy_score, 4),
    }
