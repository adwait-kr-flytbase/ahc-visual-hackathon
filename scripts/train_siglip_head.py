"""Frozen SigLIP2 + a trained temporal head. 12-way: 11 anomaly classes + normal.

Why: four of the top five genuine leaderboard entries are frozen-embedding + trained-head,
and the best (65.9) beats every fine-tuned VLM on the board. Alert-CLIP (CVPR 2026) explains
the mechanism -- CLIP's normal/abnormal TEXT embeddings are entangled, so raw similarity is a
weak discriminator and the TUNED head is what works. It is also ~100x cheaper per frame than
a 4B VLM, which makes it the cheap always-on stage of a cascade rather than a competitor.

Runs in four stages, each skippable, so a crash never costs the expensive one:
    embed  ->  train  ->  predict  ->  score
Embeddings are cached under <data-root>/emb/, so every rerun after the first is minutes.

    python scripts/train_siglip_head.py --data-root /vol --stage all
    python scripts/train_siglip_head.py --data-root /vol --stage embed --limit 20   # smoke test
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", Path("/root/src")):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from ahc_vad.taxonomy import ANOMALY_CLASSES  # noqa: E402

CLASSES = ["normal"] + sorted(ANOMALY_CLASSES)          # index 0 == normal
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


# --------------------------------------------------------------------------- frames
def sample_times(duration: float, n: int) -> list[float]:
    """Evenly spaced TIMESTAMPS, never frame indices -- clip fps ranges 1.875 to 30."""
    if duration <= 0:
        return [0.0]
    return [duration * (i + 0.5) / n for i in range(n)]


def read_frames(path: Path, times: list[float]):
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(path))
    frames = []
    for t in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames


def probe_duration(path: Path) -> float:
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        ).stdout
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- inventory
def train_clips(root: Path) -> list[dict]:
    """Every training clip, keyed on the row's class_name.

    NOT on the folder: dataset/train/wrong_way_driving/ holds 56 wrong-way rows and 108 that
    were relabelled `normal`. Keying by folder teaches normal traffic as an anomaly.
    """
    items = []
    for class_dir in sorted((root / "dataset" / "train").iterdir()):
        gt = class_dir / "ground_truth.csv"
        if not class_dir.is_dir() or not gt.exists():
            continue
        for row in csv.DictReader(gt.open(newline="", encoding="utf-8")):
            label = row["class_name"].strip()
            path = class_dir / "videos" / f"{row['video_id']}.mp4"
            if label in CLASS_TO_IDX and path.exists():
                items.append({"key": f"train/{row['video_id']}", "path": path,
                              "label": CLASS_TO_IDX[label]})
    # one row per video, deduplicated on key
    seen, out = set(), []
    for item in items:
        if item["key"] not in seen:
            seen.add(item["key"])
            out.append(item)
    return out


def eval_windows(root: Path, name: str, win: float, hop: float) -> list[dict]:
    """Sliding windows over an evaluation set, from its manifest.json."""
    from vad.windows import plan

    base = root / name
    manifest = json.loads((base / "manifest.json").read_text())["videos"]
    out = []
    for entry in manifest:
        path = base / "videos" / f"{entry['video_id']}.mp4"
        if not path.exists():
            continue
        for index, (start, end) in enumerate(plan(float(entry["duration_sec"]), win=win, hop=hop)):
            out.append({"key": f"{name}/{entry['video_id']}/w{index:04d}", "path": path,
                        "video_id": entry["video_id"], "start": start, "end": end})
    return out


# --------------------------------------------------------------------------- embed
def embed(items, model_id: str, frames_per: int, batch: int, cache: Path, device: str):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    cache.mkdir(parents=True, exist_ok=True)
    todo = [i for i in items if not (cache / f"{i['key'].replace('/', '__')}.npy").exists()]
    print(f"embed: {len(items)} items, {len(todo)} not cached", flush=True)
    if not todo:
        return

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    vision = getattr(model, "vision_model", model)

    started = time.time()
    for n, item in enumerate(todo, 1):
        if "start" in item:
            span = item["end"] - item["start"]
            times = [item["start"] + t for t in sample_times(span, frames_per)]
        else:
            times = sample_times(probe_duration(item["path"]), frames_per)
        pictures = read_frames(item["path"], times)
        if not pictures:
            continue
        with torch.no_grad():
            vectors = []
            for start in range(0, len(pictures), batch):
                inputs = processor(images=pictures[start:start + batch], return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)
                output = vision(pixel_values=pixel_values)
                pooled = getattr(output, "pooler_output", None)
                if pooled is None:
                    pooled = output.last_hidden_state.mean(dim=1)
                vectors.append(pooled.float().cpu())
            sequence = torch.cat(vectors).numpy().astype("float32")
        np.save(cache / f"{item['key'].replace('/', '__')}.npy", sequence)
        if n % 100 == 0 or n == len(todo):
            rate = n / (time.time() - started)
            print(f"  {n}/{len(todo)}  {rate:.1f}/s  eta {(len(todo)-n)/max(rate,1e-6)/60:.1f}m",
                  flush=True)


def load_cached(items, cache: Path, frames_per: int):
    import numpy as np
    keys, arrays, labels = [], [], []
    for item in items:
        path = cache / f"{item['key'].replace('/', '__')}.npy"
        if not path.exists():
            continue
        array = np.load(path)
        if array.shape[0] < frames_per:  # pad short reads by repeating the last frame
            array = np.concatenate([array, np.repeat(array[-1:], frames_per - array.shape[0], 0)])
        keys.append(item)
        arrays.append(array[:frames_per])
        labels.append(item.get("label", -1))
    return keys, np.stack(arrays) if arrays else None, np.array(labels)


# --------------------------------------------------------------------------- head
def build_head(dim: int, n_classes: int, kind: str):
    import torch.nn as nn

    class MeanPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 512), nn.GELU(),
                                     nn.Dropout(0.3), nn.Linear(512, n_classes))

        def forward(self, x):
            return self.net(x.mean(dim=1))

    class GRUHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.rnn = nn.GRU(dim, 256, batch_first=True, bidirectional=True)
            self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n_classes))

        def forward(self, x):
            h, _ = self.rnn(self.norm(x))
            return self.out(h.mean(dim=1))

    return MeanPool() if kind == "mean" else GRUHead()


def train_head(features, labels, dim, args, device):
    import numpy as np
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(labels))
    split = max(1, int(len(order) * 0.1))
    val_idx, train_idx = order[:split], order[split:]

    model = build_head(dim, len(CLASSES), args.head).to(device)
    counts = np.bincount(labels[train_idx], minlength=len(CLASSES)).astype("float32")
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train_idx)
        for start in range(0, len(train_idx), args.batch):
            chunk = train_idx[start:start + args.batch]
            optimiser.zero_grad()
            loss = criterion(model(X[chunk].to(device)), y[chunk].to(device))
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            predicted = model(X[val_idx].to(device)).argmax(1).cpu().numpy()
        accuracy = float((predicted == labels[val_idx]).mean())
        print(f"  epoch {epoch+1}/{args.epochs}  val acc {accuracy:.3f}", flush=True)
        if accuracy > best:
            best, best_state = accuracy, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    print(f"  best val acc {best:.3f}", flush=True)
    return model


# --------------------------------------------------------------------------- predict
def predict(model, items, features, device, threshold: float, single_max: float):
    """Per-window class probabilities -> merged events in the shared events.jsonl schema."""
    import torch
    from vad.merge import merge as merge_windows

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(
            model(torch.tensor(features, dtype=torch.float32).to(device)), dim=1
        ).cpu().numpy()

    per_video: dict[str, list[dict]] = {}
    for item, row in zip(items, probabilities):
        index = int(row.argmax())
        per_video.setdefault(item["video_id"], []).append({
            "start": item["start"], "end": item["end"],
            "class_name": CLASSES[index], "confidence": float(row[index]),
            "explanation": "SigLIP2 embedding with a trained temporal head over the window.",
        })

    out = {}
    for video_id, windows in per_video.items():
        keep = [w for w in windows if w["class_name"] != "normal" and w["confidence"] >= threshold]
        if not keep:
            out[video_id] = []
            continue
        span = max(w["end"] for w in windows)
        if span <= single_max:  # D1-style clip: one label, no timestamps
            best = max(keep, key=lambda w: w["confidence"])
            out[video_id] = [{"class_name": best["class_name"], "start": None, "end": None,
                              "confidence": best["confidence"],
                              "explanation": "SigLIP2 embedding + trained temporal head."}]
        else:
            # vad.merge.merge(window_events, duration, cfg=None, per_class_conf=None)
            out[video_id] = merge_windows(keep, span)
    return out


# --------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/vol"))
    parser.add_argument("--model-id", default="google/siglip2-base-patch16-224")
    parser.add_argument("--stage", default="all",
                        choices=["all", "embed", "train", "predict"])
    parser.add_argument("--eval-sets", default="synth-dev,dataset/test")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--win", type=float, default=20.0)
    parser.add_argument("--hop", type=float, default=10.0)
    parser.add_argument("--head", default="gru", choices=["mean", "gru"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--embed-batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.data_root
    cache = root / "emb"
    out_dir = args.out or (root / "out")
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = train_clips(root)
    if args.limit:
        clips = clips[: args.limit]
    print(f"train clips: {len(clips)}", flush=True)

    evaluations = {}
    for name in args.eval_sets.split(","):
        name = name.strip()
        if (root / name / "manifest.json").exists():
            windows = eval_windows(root, name, args.win, args.hop)
            if args.limit:
                windows = windows[: args.limit * 4]
            evaluations[name] = windows
            print(f"eval {name}: {len(windows)} windows", flush=True)
        else:
            print(f"eval {name}: SKIPPED, no manifest.json", flush=True)

    if args.stage in ("all", "embed"):
        embed(clips, args.model_id, args.frames, args.embed_batch, cache, args.device)
        for name, windows in evaluations.items():
            embed(windows, args.model_id, args.frames, args.embed_batch, cache, args.device)
    if args.stage == "embed":
        return 0

    import torch

    kept, features, labels = load_cached(clips, cache, args.frames)
    if features is None:
        print("no cached embeddings; run --stage embed first", file=sys.stderr)
        return 1
    print(f"training on {len(kept)} clips, dim {features.shape[-1]}", flush=True)
    model = train_head(features, labels, features.shape[-1], args, args.device)
    torch.save(model.state_dict(), out_dir / "siglip_head.pt")

    for name, windows in evaluations.items():
        kept_w, feats_w, _ = load_cached(windows, cache, args.frames)
        if feats_w is None:
            continue
        predictions = predict(model, kept_w, feats_w, args.device, args.threshold, args.win + 10)
        target = out_dir / f"siglip.{name.replace('/', '-')}.events.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for video_id, events in sorted(predictions.items()):
                handle.write(json.dumps({
                    "video_id": video_id, "events": events,
                    "runtime": {"frames_processed": args.frames, "chunks_processed": 1,
                                "end_to_end_internal_time_ms": 0, "model_runtimes": []},
                }) + "\n")
        total = sum(len(v) for v in predictions.values())
        print(f"wrote {target}: {len(predictions)} videos, {total} events", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
