"""Readers for the portal manifest and the dataset ground-truth CSVs.

Two shapes exist:
  - test/ground_truth.csv            has a `level` column
  - train/<class>/ground_truth.csv   does NOT
Both are handled; `level` is deliberately ignored here. Difficulty always comes from the
manifest, because train/ has no level information at all.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from ahc_vad.events import Event
from ahc_vad.taxonomy import NORMAL_CLASS


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    level: int
    duration_sec: float


def load_manifest(path: str | Path) -> dict[str, VideoInfo]:
    """Map video_id -> VideoInfo from the portal's manifest.json."""
    payload = json.loads(Path(path).read_text())
    return {
        entry["video_id"]: VideoInfo(
            video_id=entry["video_id"],
            level=int(entry["level"]),
            duration_sec=float(entry["duration_sec"]),
        )
        for entry in payload["videos"]
    }


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    raw = raw.strip()
    return float(raw) if raw else None


def load_ground_truth(path: str | Path) -> dict[str, list[Event]]:
    """Map video_id -> list of ground-truth Events.

    A row whose class_name is `normal` contributes no events, but the video still appears
    in the mapping with an empty list -- normal videos are scoreable false-alarm traps.
    """
    result: dict[str, list[Event]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video_id = row["video_id"].strip()
            events = result.setdefault(video_id, [])
            class_name = row["class_name"].strip()
            if class_name == NORMAL_CLASS:
                continue
            description = (row.get("description_summary") or "").strip()
            events.append(
                Event(
                    class_name=class_name,
                    start_time_sec=_parse_optional_float(row.get("start_time_sec")),
                    end_time_sec=_parse_optional_float(row.get("end_time_sec")),
                    explanation=description or None,
                )
            )
    return result
