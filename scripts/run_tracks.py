"""Detect + track objects, derive duration-defined events, emit events.jsonl.

    python scripts/run_tracks.py --videos dataset/test/videos --manifest data/manifest.json \
        --out out/track.events.jsonl

Output matches the inference lane's events.jsonl schema exactly, so it drops straight into
the shared scorer:
    {"video_id", "events": [{class_name, start, end, confidence, explanation}], "runtime": {...}}
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ahc_vad.tracks import PERSON_ID, VEHICLE_IDS, Track, derive_events

ROOT = Path(__file__).resolve().parents[1]
KEEP_CLASSES = set(VEHICLE_IDS) | {PERSON_ID}


def track_video(model, path: Path, fps: float, imgsz: int, conf: float, device: str):
    """Run tracking over one video, sampling at `fps`. Returns (tracks, times, w, h)."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(source_fps / fps))

    tracks: dict[int, Track] = {}
    times: list[float] = []
    index = 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index % stride == 0:
            ok, frame = capture.retrieve()
            if not ok:
                break
            timestamp = index / source_fps
            times.append(timestamp)
            results = model.track(
                frame, persist=True, verbose=False, imgsz=imgsz, conf=conf,
                device=device, tracker="bytetrack.yaml",
            )
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                for box, cls_id, track_id in zip(
                    boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.id.tolist()
                ):
                    cls_id = int(cls_id)
                    if cls_id not in KEEP_CLASSES:
                        continue
                    track = tracks.setdefault(int(track_id), Track(int(track_id), cls_id))
                    x0, y0, x1, y1 = box
                    track.times.append(timestamp)
                    track.centres.append(((x0 + x1) / 2, (y0 + y1) / 2))
                    track.boxes.append((x0, y0, x1, y1))
        index += 1
    capture.release()
    return list(tracks.values()), times, width, height


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    manifest = json.loads(args.manifest.read_text())["videos"]
    if args.limit:
        manifest = manifest[: args.limit]

    model = YOLO(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for entry in manifest:
            video_id = entry["video_id"]
            path = args.videos / f"{video_id}.mp4"
            if not path.exists():
                continue
            started = time.time()
            tracks, times, width, height = track_video(
                model, path, args.fps, args.imgsz, args.conf, args.device
            )
            events = derive_events(tracks, times, width, height)
            elapsed_ms = (time.time() - started) * 1000
            handle.write(json.dumps({
                "video_id": video_id,
                "events": events,
                "runtime": {
                    "frames_processed": len(times),
                    "chunks_processed": 1,
                    "end_to_end_internal_time_ms": round(elapsed_ms, 1),
                    "model_runtimes": [{
                        "model_name": args.model, "call_count": len(times),
                        "total_time_ms": round(elapsed_ms, 1),
                        "average_time_ms": round(elapsed_ms / max(1, len(times)), 2),
                    }],
                },
            }) + "\n")
            handle.flush()
            written += 1
            print(f"  {video_id}: {len(tracks):3d} tracks, {len(times):4d} frames, "
                  f"{len(events)} events, {elapsed_ms/1000:.1f}s", flush=True)
    print(f"\nwrote {args.out}: {written} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
