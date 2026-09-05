"""Build the demo review page from a run's events file.

Renders one self-contained HTML file: the video, the ground truth and the model's
predictions on a shared time axis, and the verdict on every span.

Matching is NOT reimplemented here -- it calls ahc_vad.scoring.match_events with the same
policies ahc_vad.scoring.score uses, so the page can never disagree with the scorer.

    python demo/build.py --run gemini
    open demo/index.html

Data is inlined rather than fetched because file:// blocks XHR, and a venue demo must not
depend on a local server or on wifi.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ahc_vad.events import Event, temporal_iou  # noqa: E402
from ahc_vad.groundtruth import load_ground_truth, load_manifest  # noqa: E402
from ahc_vad.scoring import DIFFICULTY_MARKS, MatchPolicy, match_events  # noqa: E402
from ahc_vad.taxonomy import ANOMALY_CLASSES  # noqa: E402

POLICY_D1 = MatchPolicy(require_temporal=False)
POLICY_D23 = MatchPolicy()


def under_root(path: Path) -> str:
    """Path relative to the repo root when it is inside it, else as given."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_predictions(paths: list[Path]) -> tuple[dict[str, dict], list[str]]:
    """Read one or more <run>.events.jsonl -> {video_id: raw row}, plus a list of complaints.

    Several files because D1 and D2/D3 are usually run separately; passing both gives one page
    over the whole test set. Malformed rows are reported, never silently dropped: a demo that
    quietly hides bad model output is the same failure mode as a scorer that does.
    """
    rows: dict[str, dict] = {}
    problems: list[str] = []
    for path in paths:
        _read_one(path, rows, problems)
    return rows, problems


def _read_one(path: Path, rows: dict[str, dict], problems: list[str]) -> None:
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name} line {number}: not JSON ({exc.msg})")
            continue
        video_id = row.get("video_id")
        if not video_id:
            problems.append(f"{path.name} line {number}: no video_id")
            continue
        if video_id in rows:
            problems.append(f"{path.name} line {number}: {video_id} already seen, keeping the last")
        rows[video_id] = row


def load_windows(paths: list[Path]) -> dict[str, dict]:
    """Read the <run>.windows.jsonl beside each events file, when it exists.

    It carries the window grid the system actually processed -- `windows` holds detections (one
    window can emit several) and `empty_window_raw` holds the windows that returned nothing, so
    the distinct union of their `window` fields is the grid. Verified against `chunks_processed`
    on all 34 videos.
    """
    rows: dict[str, dict] = {}
    for path in paths:
        sibling = path.with_name(path.name.replace(".events.jsonl", ".windows.jsonl"))
        if sibling == path or not sibling.exists():
            continue
        for line in sibling.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["video_id"]] = row
    return rows


def window_grid(row: dict | None) -> list[list[float]]:
    """The distinct [start, end] windows the model was run on, in completion order."""
    if not row:
        return []
    seen = {tuple(w["window"]) for w in row.get("windows", []) if w.get("window")}
    seen |= {tuple(w["window"]) for w in row.get("empty_window_raw", []) if w.get("window")}
    return [list(w) for w in sorted(seen, key=lambda w: (w[1], w[0]))]


def timing(runtime: dict) -> dict:
    """Measured per-window latency, and whether it keeps up with the window hop.

    Only run-level aggregates were recorded, not per-window times, so the replay spaces alerts by
    the run's mean. The page says so rather than implying we timed each window.
    """
    models = (runtime or {}).get("model_runtimes") or []
    if not models:
        return {}
    m = models[0]
    return {"model_name": m.get("model_name"), "calls": m.get("call_count"),
            "mean_ms": m.get("average_time_ms"), "p50_ms": m.get("p50_time_ms"),
            "p95_ms": m.get("p95_time_ms"), "max_ms": m.get("max_time_ms"),
            "frames": (runtime or {}).get("frames_processed"),
            "chunks": (runtime or {}).get("chunks_processed"),
            "end_to_end_ms": (runtime or {}).get("end_to_end_internal_time_ms")}


def to_events(raw_events: list[dict], video_id: str, problems: list[str]) -> tuple[list[Event], list[dict]]:
    """Predicted dicts -> (Events the scorer accepts, display records).

    A prediction the scorer cannot represent -- unknown class, inverted span -- is kept for
    display and marked rejected, so it is visible on the page instead of vanishing.
    """
    events: list[Event] = []
    display: list[dict] = []
    for raw in raw_events:
        class_name = (raw.get("class_name") or "").strip()
        start, end = raw.get("start"), raw.get("end")
        record = {
            "class_name": class_name or "(missing)",
            "start": start,
            "end": end,
            "confidence": raw.get("confidence"),
            "explanation": (raw.get("explanation") or "").strip(),
            "window": raw.get("window"),
            "rejected": None,
        }
        if class_name not in ANOMALY_CLASSES:
            record["rejected"] = f"{class_name!r} is not one of the 11 submittable classes"
            problems.append(f"{video_id}: {record['rejected']}")
            display.append(record)
            continue
        try:
            events.append(Event(class_name, start, end))
        except ValueError as exc:
            record["rejected"] = str(exc)
            problems.append(f"{video_id}: {exc}")
            display.append(record)
            continue
        record["index"] = len(events) - 1
        display.append(record)
    return events, display


def apply_top1(row: dict) -> dict:
    """Keep only the highest-confidence event. Sound at D1, where a video has at most one truth
    event; at D2/D3 it discards real detections, so the page labels which policy produced it."""
    events = row.get("events") or []
    if len(events) <= 1:
        return row
    best = max(events, key=lambda e: e.get("confidence") or 0)
    return {**row, "events": [best]}


def build_video(video_id, info, gt_events, prediction, problems, window_row=None):
    """One video's payload: both lanes, every span already carrying its verdict.

    Three states, never collapsed: no row at all is `not_run`, a row carrying `failed` is a
    video the run broke on, and anything else is a real prediction. Only the last is scored --
    counting a broken video as a correct normal flatters every partial run.
    """
    failed = (prediction or {}).get("failed")
    state = "not_run" if prediction is None else ("failed" if failed else "ok")
    if failed:
        problems.append(f"{video_id}: the run failed on this video -- {failed}")
    predicted = state == "ok"
    pred_events, pred_display = to_events(
        prediction.get("events", []) if predicted else [], video_id, problems
    )
    policy = POLICY_D1 if info.level == 1 else POLICY_D23
    result = match_events(gt_events, pred_events, policy) if predicted else None

    gt_to_pred = {gi: (pi, iou) for gi, pi, iou in result.matched} if result else {}
    pred_to_gt = {pi: (gi, iou) for gi, pi, iou in result.matched} if result else {}

    truth = []
    for index, event in enumerate(gt_events):
        pair = gt_to_pred.get(index)
        truth.append({
            "class_name": event.class_name,
            "start": event.start_time_sec if event.is_localised else 0.0,
            "end": event.end_time_sec if event.is_localised else info.duration_sec,
            "localised": event.is_localised,
            "description": event.explanation or "",
            # Nothing was predicted for this video yet, so it is not a miss -- it is unscored.
            "verdict": ("hit" if pair else "miss") if predicted else "pending",
            "iou": pair[1] if pair else None,
            "pair": pair[0] if pair else None,
        })

    model = []
    for record in pred_display:
        index = record.get("index")
        pair = pred_to_gt.get(index) if index is not None else None
        localised = record["start"] is not None and record["end"] is not None
        model.append({
            "class_name": record["class_name"],
            "start": record["start"] if localised else 0.0,
            "end": record["end"] if localised else info.duration_sec,
            "localised": localised,
            "confidence": record["confidence"],
            "explanation": record["explanation"],
            "rejected": record["rejected"],
            "from_window": record.get("window"),
            "verdict": "hit" if pair else "false_alarm",
            "iou": pair[1] if pair else None,
            "pair": pair[0] if pair else None,
            # Shown at D1, where the scorer ignores timing: says what the overlap would have
            # been if this difficulty were timed. Context, never a score.
            "iou_untimed": (
                round(temporal_iou((record["start"], record["end"]),
                                   (truth[pair[0]]["start"], truth[pair[0]]["end"])), 3)
                if pair and info.level == 1 and localised else None
            ),
        })

    return {
        "id": video_id,
        "level": info.level,
        "duration": info.duration_sec,
        "src": f"../dataset/test/videos/{video_id}.mp4",
        "predicted": predicted,
        "state": state,
        "failed": failed,
        "truth": truth,
        "model": model,
        "runtime": (prediction or {}).get("runtime", {}),
        "timing": timing((prediction or {}).get("runtime", {})),
        "grid": window_grid(window_row) if predicted else [],
        "tally": {
            "hits": result.true_positives if result else 0,
            "misses": result.misses if result else 0,
            "false_alarms": result.false_alarms if result else 0,
        },
    }


def throughput(videos):
    """Run-level cost and whether inference keeps up with the window hop.

    The hop is how often a new window closes, so a system keeps up in real time when a window is
    scored in less than one hop. Hop is read off the grid, not assumed.
    """
    hops, means, p95s, frames, chunks = [], [], [], 0, 0
    model = None
    for video in videos:
        t = video.get("timing") or {}
        if not t.get("mean_ms"):
            continue
        model = model or t.get("model_name")
        frames += t.get("frames") or 0
        chunks += t.get("chunks") or 0
        # weight by call count: a mean of per-video means over-weights the short D1 videos,
        # which are one window each
        means.append((t["mean_ms"], t.get("calls") or 1))
        if t.get("p95_ms"):
            p95s.append((t["p95_ms"], t.get("calls") or 1))
        grid = video.get("grid") or []
        if len(grid) > 1:
            hops.append(round(grid[1][0] - grid[0][0], 3))
    if not means:
        return {}
    hop = max(set(hops), key=hops.count) if hops else None
    mean_ms = sum(v * n for v, n in means) / sum(n for _, n in means)
    p95_ms = (sum(v * n for v, n in p95s) / sum(n for _, n in p95s)) if p95s else None
    return {"model_name": model, "frames": frames, "windows": chunks, "hop_sec": hop,
            "mean_ms": mean_ms, "p95_ms": p95_ms,
            "realtime_mean": (hop * 1000 / mean_ms) if hop and mean_ms else None,
            "realtime_p95": (hop * 1000 / p95_ms) if hop and p95_ms else None}


def summarise(videos):
    """Per-difficulty totals over the videos that actually have a prediction."""
    levels = {}
    for video in videos:
        if not video["predicted"]:
            continue
        entry = levels.setdefault(video["level"], {"hits": 0, "misses": 0, "false_alarms": 0, "videos": 0})
        entry["videos"] += 1
        for key in ("hits", "misses", "false_alarms"):
            entry[key] += video["tally"][key]
    out = []
    for level in sorted(DIFFICULTY_MARKS):
        entry = levels.get(level, {"hits": 0, "misses": 0, "false_alarms": 0, "videos": 0})
        found, fa, missed = entry["hits"], entry["false_alarms"], entry["misses"]
        precision = found / (found + fa) if (found + fa) else None
        recall = found / (found + missed) if (found + missed) else None
        out.append({
            "level": level,
            "marks": DIFFICULTY_MARKS[level],
            "videos_scored": entry["videos"],
            "videos_total": sum(1 for v in videos if v["level"] == level),
            **entry,
            "precision": precision,
            "recall": recall,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="gemini", help="run name; reads out/<run>.events.jsonl")
    parser.add_argument("--events", type=Path, nargs="+",
                        help="explicit events jsonl files, overrides --run; several are merged")
    parser.add_argument("--name", help="run label shown in the page header")
    parser.add_argument("--top1", action="store_true",
                        help="keep only the highest-confidence event per video, as the D1 scoring "
                             "policy does. Changes the false-alarm count, so the page says which "
                             "policy it drew.")
    parser.add_argument("--gt", type=Path, default=ROOT / "dataset/test/ground_truth.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "demo/index.html")
    args = parser.parse_args()

    events_paths = args.events or [ROOT / "out" / f"{args.run}.events.jsonl"]
    for path in events_paths:
        if not path.exists():
            parser.error(f"no events file at {path}")

    manifest = load_manifest(args.manifest)
    ground_truth = load_ground_truth(args.gt)
    predictions, problems = load_predictions(events_paths)
    if args.top1:
        predictions = {vid: apply_top1(row) for vid, row in predictions.items()}

    unknown = sorted(set(predictions) - set(manifest))
    problems += [f"{video_id}: predicted but not in the manifest, ignored" for video_id in unknown]

    windows = load_windows(events_paths)
    videos = [
        build_video(video_id, info, ground_truth.get(video_id, []), predictions.get(video_id),
                    problems, windows.get(video_id))
        for video_id, info in manifest.items()
    ]

    payload = {
        "run": {
            "name": args.name or (args.run if not args.events
                                  else " + ".join(p.stem.replace(".events", "") for p in events_paths)),
            "events_path": "  +  ".join(under_root(p) for p in events_paths),
            "built_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
            "policy": "top-1 event per video" if args.top1 else "every predicted event",
            "videos_predicted": sum(1 for v in videos if v["state"] == "ok"),
            "videos_failed": sum(1 for v in videos if v["state"] == "failed"),
            "videos_not_run": sum(1 for v in videos if v["state"] == "not_run"),
            "videos_total": len(videos),
        },
        "throughput": throughput(videos),
        "problems": problems,
        "videos": videos,
        "summary": summarise(videos),
    }

    template = (Path(__file__).parent / "template.html").read_text()
    # </script> inside the JSON would close the tag early.
    blob = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    args.out.write_text(template.replace("__DATA__", blob))

    run = payload["run"]
    print(f"wrote {under_root(args.out)}  ({run['videos_predicted']}/{run['videos_total']} predicted, "
          f"{run['videos_failed']} failed, {run['videos_not_run']} not run)")
    for problem in problems:
        print(f"  ! {problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
