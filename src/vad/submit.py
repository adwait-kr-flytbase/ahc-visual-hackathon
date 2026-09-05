"""events.jsonl -> portal submission JSON.

Thin adapter over ahc_vad.submission (the canonical builder/validator). This module only
translates the inference lane's jsonl into ahc_vad Events and applies the D1 top-k policy.

  PYTHONPATH=src python -m vad.submit --events out/sft.best.jsonl \
      --manifest data/manifest.json --out out/submission.json --model-name qwen3vl4b-lora
"""
import argparse, json

from ahc_vad.events import Event
from ahc_vad.groundtruth import load_manifest
from ahc_vad.submission import (
    EXPLANATION_MIN_CHARS, RuntimeMetadata, build_submission, validate_submission,
    write_submission,
)


def to_events(records, manifest, d1_topk=1):
    """jsonl records -> ({video_id: [Event]}, {video_id: RuntimeMetadata}, total_ms)."""
    preds, runtimes, total_ms = {}, {}, 0.0
    for vid, d in records.items():
        info = manifest.get(vid)
        if info is None:
            continue
        dur = info.duration_sec
        evs = sorted(d.get("events", []), key=lambda e: -e.get("confidence", 0.5))
        if info.level == 1:
            evs = evs[:d1_topk]
        out = []
        for e in evs:
            expl = (e.get("explanation") or "").strip()
            expl = expl if len(expl) >= EXPLANATION_MIN_CHARS else None
            if info.level == 1:
                out.append(Event(e["class_name"], None, None, expl))
                continue
            s = max(0.0, float(e["start"]))
            t = min(float(e["end"]), dur)
            s = min(s, max(0.0, dur - 0.5))
            if t <= s:
                t = min(dur, s + 0.5)
            out.append(Event(e["class_name"], round(s, 2), round(t, 2), expl))
        preds[vid] = out
        rt = d.get("runtime") or {}
        runtimes[vid] = RuntimeMetadata(
            frames_processed=rt.get("frames_processed", 0),
            chunks_processed=rt.get("chunks_processed", 1),
            end_to_end_internal_time_ms=rt.get("end_to_end_internal_time_ms", 0.0),
            model_runtimes=rt.get("model_runtimes", []),
        )
        total_ms += runtimes[vid].end_to_end_internal_time_ms
    return preds, runtimes, total_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="my-model")
    ap.add_argument("--submission-id", default="run-01")
    ap.add_argument("--hardware", default="unspecified")
    ap.add_argument("--d1-topk", type=int, default=1,
                    help="max events kept per level-1 video (gt has exactly one)")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    records = {}
    for line in open(args.events):
        line = line.strip()
        if line:
            d = json.loads(line)
            records[d["video_id"]] = d

    preds, runtimes, total_ms = to_events(records, manifest, args.d1_topk)
    payload = build_submission(preds, manifest, submission_id=args.submission_id,
                               model_name=args.model_name, runtimes=runtimes,
                               hardware=args.hardware, total_wall_time_ms=total_ms)
    problems = validate_submission(payload, manifest)
    if problems:
        for p in problems[:20]:
            print("INVALID:", p)
        raise SystemExit(f"{len(problems)} validation problems; nothing written")
    write_submission(payload, args.out)
    n = sum(len(p["events"]) for p in payload["predictions"])
    print(f"wrote {args.out}: {len(payload['predictions'])} videos, {n} events, valid")


if __name__ == "__main__":
    main()
