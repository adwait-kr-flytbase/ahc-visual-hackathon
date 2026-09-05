"""Re-merge raw window predictions under many configs and score each. No GPU, no re-inference.

Scored by ahc_vad.scoring via ahc_vad.compat -- best-IoU-first matching, which is the correct
behaviour when several predictions overlap one truth event (exactly the regime this sweep explores).

  python -m vad.sweep --windows out/zeroshot.windows.jsonl \
      --gt dataset/test/ground_truth.csv --manifest .context/artifacts/manifest.json
"""
import argparse, itertools, json, os, tempfile

from ahc_vad.compat import score_events_jsonl as score  # canonical matcher (best-IoU-first)

from .merge import merge


def apply_cfg(windows_path, cfg, out_path):
    with open(out_path, "w") as f:
        for line in open(windows_path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            evs = merge(d["windows"], d["duration"], cfg=cfg)
            f.write(json.dumps({"video_id": d["video_id"], "events": evs,
                                "runtime": d.get("runtime", {})}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--best-out", default=None, help="write the winning events.jsonl here")
    ap.add_argument("--metric", default="f1", choices=["f1", "precision", "recall"])
    args = ap.parse_args()

    grid = {
        "min_conf": [0.0, 0.25, 0.35, 0.5, 0.65, 0.8],
        "gap_tol":  [0.0, 3.0, 6.0, 12.0],
        "min_dur":  [0.0, 1.0, 3.0],
    }
    keys = list(grid)
    rows = []
    tmp = tempfile.mktemp(suffix=".jsonl")
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        apply_cfg(args.windows, cfg, tmp)
        s = score(tmp, args.gt, args.manifest)
        rows.append((s["overall"][args.metric], cfg, s))

    rows.sort(key=lambda r: -r[0])
    print(f"\ntop 10 by overall {args.metric}:")
    for v, cfg, s in rows[:10]:
        lv = " ".join(f"{k}:P{d['precision']:.2f}/R{d['recall']:.2f}/FA{d['fa']}"
                      for k, d in s["levels"].items())
        print(f"  {v:.4f}  {cfg}   {lv}")

    best_v, best_cfg, best_s = rows[0]
    print("\nbest config:", json.dumps(best_cfg))
    print(json.dumps(best_s["levels"], indent=2))
    print("\nper class (best config):")
    for c, d in best_s["classes"].items():
        print(f"  {c:36s} found {d['found']:3d}  fa {d['fa']:3d}  missed {d['missed']:3d}")

    if args.best_out:
        apply_cfg(args.windows, best_cfg, args.best_out)
        print("\nwrote", args.best_out)
    os.path.exists(tmp) and os.remove(tmp)


if __name__ == "__main__":
    main()
