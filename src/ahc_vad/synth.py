"""Compose long, multi-event videos from the short training clips.

train/ has no long or multi-event videos, but D2+D3 is 75 of the 100 marks. This
manufactures that training signal: lay down normal filler, splice in anomaly clips at
chosen offsets, and record exactly where each one went.

Three measured facts drive the design (.context/09-dataset-profile.md):

1. For 7 of 11 classes the labelled event IS the whole source clip. Naive concatenation
   therefore makes event onset identical to a scene cut, and a model learns to detect
   cuts. Mitigation: the normal filler is chopped into pieces from DIFFERENT clips at a
   rate matched to event density, so a cut carries no information about the label.

2. Clip duration leaks the class -- waterlogging is 95 clips at ~5.7 s, loitering is 300
   clips with 3 distinct durations. Mitigation: every segment is trimmed to a random
   sub-span, so event length varies independently of source length.

3. Train is 85.5% 1280x720/15fps but the test pack is 640x640, 720x404, 896x448, 256x192
   at 24/30/1.875 fps. Mitigation: each output samples a (resolution, fps) profile from
   the TEST-side distribution, and letterboxes rather than stretches.
"""

import csv
import json
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# (width, height, fps) profiles observed in dataset/test. Weighted by how often each
# appears across the 34 public videos.
TEST_PROFILES = [
    ((640, 640), 24.0, 11),
    ((720, 404), 30.0, 4),
    ((896, 448), 15.0, 6),
    ((1280, 720), 25.0, 5),
    ((800, 410), 30.0, 2),
    ((256, 192), 30.0, 2),
    ((640, 360), 30.0, 2),
]


@dataclass(frozen=True)
class Segment:
    """One trimmed piece of a source clip, placed on the output timeline."""

    source: Path
    source_start: float
    duration: float
    class_name: str | None  # None -> normal filler


@dataclass
class Composition:
    video_id: str
    level: int
    segments: list[Segment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def events(self) -> list[dict]:
        """Absolute-time events on the output timeline, in order."""
        out, t = [], 0.0
        for segment in self.segments:
            if segment.class_name is not None:
                out.append({
                    "class_name": segment.class_name,
                    "start": round(t, 3),
                    "end": round(t + segment.duration, 3),
                })
            t += segment.duration
        return out


def _load_real_durations(meta_tsv: Path) -> dict[str, float]:
    """video_id -> true clip duration, from the ffprobe sweep in .context/artifacts."""
    durations: dict[str, float] = {}
    if not meta_tsv.exists():
        return durations
    with meta_tsv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                durations[row["video_id"]] = float(row["duration"])
            except (KeyError, TypeError, ValueError):
                continue
    return durations


def load_clip_pool(dataset_root: Path, meta_tsv: Path | None = None) -> dict[str, list[dict]]:
    """Map class_name -> clips, each {path, real_duration, event_start, event_end}.

    `real_duration` is the clip's true length. Requesting more than that from ffmpeg
    silently yields a shorter segment, which desynchronises every downstream timestamp --
    so it is required, and clips without it are dropped.
    """
    meta_tsv = meta_tsv or (dataset_root.parent / ".context" / "artifacts" / "videometa.tsv")
    real = _load_real_durations(meta_tsv)
    pool: dict[str, list[dict]] = {}
    for class_dir in sorted((dataset_root / "train").iterdir()):
        if not class_dir.is_dir():
            continue
        gt_path = class_dir / "ground_truth.csv"
        if not gt_path.exists():
            continue
        rows = list(csv.DictReader(gt_path.open(newline="", encoding="utf-8")))
        clips = []
        for row in rows:
            path = class_dir / "videos" / f"{row['video_id']}.mp4"
            if not path.exists():
                continue
            duration = real.get(row["video_id"])
            if duration is None or duration < 1.0:
                continue
            start = (row.get("start_time_sec") or "").strip()
            end = (row.get("end_time_sec") or "").strip()
            clips.append({
                "path": path,
                "real_duration": duration,
                "event_start": float(start) if start else None,
                "event_end": float(end) if end else None,
                "description": (row.get("description_summary") or "").strip(),
            })
        if clips:
            pool[class_dir.name] = clips
    return pool


def _trim_window(clip: dict, want: float, rng: random.Random) -> tuple[float, float]:
    """Pick (source_start, duration) inside a clip, targeting `want` seconds.

    Always clamped to the clip's real length: ffmpeg returns a SHORTER segment when asked
    for more than exists, with no error, which would desynchronise every later timestamp.

    When the clip carries a real sub-span event, stay inside it so the label stays true.
    Otherwise the whole clip is the event, so any sub-span is still that class.
    """
    real = clip["real_duration"]
    start = clip["event_start"] if clip["event_start"] is not None else 0.0
    start = max(0.0, min(start, max(0.0, real - 0.5)))
    end = clip["event_end"]
    if end is None or end <= start:
        end = real
    end = min(end, real)

    span = max(0.5, end - start)
    duration = max(0.5, min(want, span))
    slack = span - duration
    offset = rng.uniform(0, slack) if slack > 0.1 else 0.0
    source_start = round(start + offset, 3)
    duration = round(min(duration, real - source_start), 3)
    return source_start, max(0.5, duration)


def compose(
    video_id: str,
    level: int,
    target_duration: float,
    class_names: list[str],
    pool: dict[str, list[dict]],
    rng: random.Random,
    *,
    event_duration_range: tuple[float, float] = (4.0, 60.0),
    filler_cut_range: tuple[float, float] = (8.0, 45.0),
) -> Composition:
    """Lay out one long video: filler, event, filler, event, ... filler.

    The filler between events is deliberately built from SEVERAL different normal clips so
    that splices appear far more often than events do. Without that, "a cut happened" is a
    perfect predictor of "an event started".
    """
    normals = pool["normal"]
    segments: list[Segment] = []
    remaining = target_duration

    # Reserve time for the events, spend the rest on filler.
    planned: list[tuple[str, float]] = []
    for name in class_names:
        low, high = event_duration_range
        planned.append((name, rng.uniform(low, min(high, max(low, remaining / max(1, len(class_names)))))))
    event_total = sum(d for _, d in planned)
    filler_total = max(0.0, target_duration - event_total)
    gaps = len(planned) + 1
    per_gap = filler_total / gaps if gaps else 0.0

    def add_filler(budget: float) -> None:
        """Fill `budget` seconds using multiple normal clips -> multiple distractor cuts."""
        left = budget
        guard = 0
        while left > 1.0 and guard < 400:
            guard += 1
            clip = rng.choice(normals)
            piece = min(left, rng.uniform(*filler_cut_range))
            source_start, duration = _trim_window(
                {"real_duration": clip["real_duration"], "event_start": None, "event_end": None},
                piece, rng,
            )
            segments.append(Segment(clip["path"], source_start, duration, None))
            left -= duration

    for name, want in planned:
        add_filler(per_gap)
        clip = rng.choice(pool[name])
        source_start, duration = _trim_window(clip, want, rng)
        segments.append(Segment(clip["path"], source_start, duration, name))
    add_filler(per_gap)

    # Events are clamped to their source clip's real length, so the planned event budget
    # over-estimates and the video lands short. Top up with filler to hit the target --
    # D2 in the public pack is always exactly 240 s and the synthetic set should match.
    shortfall = target_duration - sum(seg.duration for seg in segments)
    if shortfall > 1.0:
        add_filler(shortfall)

    return Composition(video_id=video_id, level=level, segments=segments)


def render(
    composition: Composition,
    out_path: Path,
    profile: tuple[tuple[int, int], float, int],
    *,
    crf: int = 30,
    preset: str = "ultrafast",
    timeout: int = 900,
) -> bool:
    """Render with one ffmpeg pass: trim -> scale -> pad -> fps -> concat.

    Every segment is normalised to identical geometry and frame rate before concat,
    because the source pool spans 23 resolutions and 10 frame rates and the concat
    demuxer silently corrupts output when codec parameters differ.
    """
    (width, height), fps, _ = profile
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for segment in composition.segments:
        cmd += ["-ss", f"{segment.source_start}", "-t", f"{segment.duration}", "-i", str(segment.source)]

    chains, labels = [], []
    for index in range(len(composition.segments)):
        chains.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={fps},setsar=1,format=yuv420p[v{index}]"
        )
        labels.append(f"[v{index}]")
    graph = ";".join(chains) + ";" + "".join(labels) + f"concat=n={len(labels)}:v=1:a=0[out]"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd += [
        "-filter_complex", graph, "-map", "[out]",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-an", str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and out_path.exists()


def write_dataset(
    compositions: list[Composition],
    out_dir: Path,
    durations: dict[str, float],
) -> None:
    """Emit ground_truth.csv and manifest.json in exactly the dataset/test shape."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ground_truth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "video_id", "level", "is_anomaly", "class_name",
            "start_time_sec", "end_time_sec", "description_summary",
        ])
        for composition in compositions:
            events = composition.events()
            if not events:
                writer.writerow([composition.video_id, composition.level, "false", "normal", "", "", ""])
                continue
            for event in events:
                writer.writerow([
                    composition.video_id, composition.level, "true", event["class_name"],
                    f"{event['start']:.3f}", f"{event['end']:.3f}", "",
                ])
    with (out_dir / "videos.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "filename"])
        for composition in compositions:
            writer.writerow([composition.video_id, f"videos/{composition.video_id}.mp4"])
    manifest = {
        "schema_version": "1.0",
        "videos": [
            {
                "video_id": c.video_id,
                "level": c.level,
                "domain": "",
                "duration_sec": round(durations.get(c.video_id, c.duration), 3),
            }
            for c in compositions
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
