"""Inference orchestrator.

Writes TWO files:
  windows.jsonl -- raw per-window predictions. sweep.py re-merges these for free.
  events.jsonl  -- merged video-level events (the shared contract format).

  python -m vad.run --videos dataset/test/videos --manifest .context/artifacts/manifest.json \
      --engine hf --model Qwen/Qwen3-VL-4B-Instruct --out out/zeroshot
"""
import argparse, json, os, glob
from concurrent.futures import ThreadPoolExecutor

from . import frames as F
from . import windows as W
from . import prompts as P
from .merge import merge
from .runtime import Timer


def video_list(videos_dir, manifest):
    if manifest and os.path.exists(manifest):
        man = {v["video_id"]: v for v in json.load(open(manifest))["videos"]}
    else:
        man = {}
    out = []
    for path in sorted(glob.glob(os.path.join(videos_dir, "*.mp4"))):
        vid = os.path.splitext(os.path.basename(path))[0]
        info = man.get(vid, {})
        dur = info.get("duration_sec") or F.probe(path)["duration"]
        out.append({"video_id": vid, "path": path, "duration": float(dur),
                    "level": info.get("level", 1)})
    return out


def run_video(v, engine, args, timer):
    wins = W.plan(v["duration"], win=args.win, hop=args.hop, single_max=args.single_max)
    timer.chunks = len(wins)
    raws = []

    def one(w):
        t0, t1 = w
        fr, _ = F.sample(v["path"], t0, t1, n=args.frames, max_side=args.max_side)
        if not fr:
            return []
        span = t1 - t0
        with timer.track(engine.name):
            text = engine.generate(fr, P.SYSTEM, P.user_prompt(span))
        evs = P.parse_response(text, span)
        timer.frames += len(fr)
        if not evs and args.keep_raw:      # so an empty result is distinguishable from a parse failure
            raws.append({"window": [t0, t1], "raw": (text or "")[:600]})
        for e in evs:                                  # window-relative -> absolute
            e["start"] = round(min(v["duration"], t0 + e["start"]), 2)
            e["end"] = round(min(v["duration"], t0 + e["end"]), 2)
            e["window"] = [t0, t1]
        return evs

    if engine.concurrent and args.workers > 1 and len(wins) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            got = list(ex.map(one, wins))
    else:
        got = [one(w) for w in wins]
    return [e for sub in got for e in sub], raws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True, help="output prefix, e.g. out/zeroshot")
    ap.add_argument("--engine", choices=["hf", "server", "gemini"], default="hf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--win", type=float, default=20.0)
    ap.add_argument("--hop", type=float, default=10.0)
    ap.add_argument("--single-max", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N videos")
    ap.add_argument("--keep-raw", action="store_true", default=True,
                    help="record model text for windows that produced no events")
    args = ap.parse_args()

    from .engine import GeminiEngine, HFEngine, ServerEngine
    if args.engine == "hf":
        engine = HFEngine(args.model, adapter=args.adapter)
    elif args.engine == "gemini":
        engine = GeminiEngine(args.model)
    else:
        engine = ServerEngine(args.model, base_url=args.base_url)

    vids = video_list(args.videos, args.manifest)
    if args.skip:
        vids = vids[args.skip:]
    if args.limit:
        vids = vids[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fw = open(args.out + ".windows.jsonl", "w")
    fe = open(args.out + ".events.jsonl", "w")
    failures = []
    for i, v in enumerate(vids, 1):
        timer = Timer()
        failed = None
        try:
            raw, empties = run_video(v, engine, args, timer)
        except Exception as exc:           # never lose 33 videos to one bad response
            failed = f"{type(exc).__name__}: {exc}"
            failures.append((v["video_id"], failed))
            print(f"[{i}/{len(vids)}] {v['video_id']} FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            raw, empties = [], []
        meta = timer.metadata()
        row = {"video_id": v["video_id"], "duration": v["duration"],
               "level": v["level"], "windows": raw,
               "empty_window_raw": empties, "runtime": meta}
        if failed:
            row["failed"] = failed          # a FAILED video must never look like a NORMAL one
        fw.write(json.dumps(row) + "\n")
        fw.flush()
        events = merge(raw, v["duration"])
        erow = {"video_id": v["video_id"], "events": events, "runtime": meta}
        if failed:
            erow["failed"] = failed
        fe.write(json.dumps(erow) + "\n")
        fe.flush()
        print(f"[{i}/{len(vids)}] {v['video_id']} L{v['level']} {v['duration']:.0f}s "
              f"{timer.chunks}w -> {len(raw)} raw -> {len(events)} events "
              f"({meta['end_to_end_internal_time_ms']}ms)", flush=True)
    fw.close(); fe.close()
    if failures:
        print(f"\n{len(failures)} video(s) failed:", flush=True)
        for vid, msg in failures:
            print(f"  {vid}: {msg}", flush=True)
    print("wrote", args.out + ".windows.jsonl", "and", args.out + ".events.jsonl")


if __name__ == "__main__":
    main()
