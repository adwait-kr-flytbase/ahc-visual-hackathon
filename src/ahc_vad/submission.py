"""Build and validate the portal submission payload.

Shape is copied from data/submission-template.json. See .context/07-platform-and-scoring.md.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from ahc_vad.events import Event
from ahc_vad.groundtruth import VideoInfo
from ahc_vad.taxonomy import is_valid_anomaly_class

SCHEMA_VERSION = "1.0"
EXPLANATION_MIN_CHARS = 20
EXPLANATION_MAX_CHARS = 500


@dataclass
class RuntimeMetadata:
    """Self-reported per-video timings. Feeds the latency bonus -- report honestly."""

    frames_processed: int = 0
    chunks_processed: int = 1
    end_to_end_internal_time_ms: float = 0.0
    model_runtimes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "chunks_processed": self.chunks_processed,
            "end_to_end_internal_time_ms": self.end_to_end_internal_time_ms,
            "model_runtimes": list(self.model_runtimes),
        }


def _event_to_dict(event: Event) -> dict:
    payload = {
        "class_name": event.class_name,
        "start_time_sec": event.start_time_sec,
        "end_time_sec": event.end_time_sec,
    }
    if event.explanation:
        payload["explanation"] = event.explanation
    return payload


def build_submission(
    predictions: dict[str, list[Event]],
    manifest: dict[str, VideoInfo],
    *,
    submission_id: str,
    model_name: str,
    runtimes: dict[str, RuntimeMetadata] | None = None,
    hardware: str = "unspecified",
    total_wall_time_ms: float = 0.0,
) -> dict:
    """Build the full payload. Videos absent from `predictions` get an empty events list.

    Video order follows the manifest, so output is byte-stable across runs.
    """
    runtimes = runtimes or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "submission_id": submission_id,
        "model_name": model_name,
        "run_metadata": {
            "total_wall_time_ms": total_wall_time_ms,
            "max_parallel_videos": 1,
            "hardware": hardware,
        },
        "predictions": [
            {
                "video_id": video_id,
                "events": [_event_to_dict(e) for e in predictions.get(video_id, [])],
                "runtime_metadata": runtimes.get(video_id, RuntimeMetadata()).to_dict(),
            }
            for video_id in manifest
        ],
    }


def validate_submission(payload: dict, manifest: dict[str, VideoInfo]) -> list[str]:
    """Return a list of human-readable problems. Empty list means the payload is valid."""
    problems: list[str] = []
    seen: set[str] = set()

    for prediction in payload.get("predictions", []):
        video_id = prediction.get("video_id")
        if video_id not in manifest:
            problems.append(f"{video_id}: not a known video id")
            continue
        if video_id in seen:
            problems.append(f"{video_id}: duplicate prediction entry")
            continue
        seen.add(video_id)

        info = manifest[video_id]
        for event in prediction.get("events", []):
            name = event.get("class_name")
            if not is_valid_anomaly_class(name):
                problems.append(f"{video_id}: {name!r} is not one of the 11 submittable classes")
            start, end = event.get("start_time_sec"), event.get("end_time_sec")
            if info.level == 1:
                if start is not None or end is not None:
                    problems.append(f"{video_id}: D1 events must have null start and end times")
            else:
                if start is None or end is None:
                    problems.append(f"{video_id}: D{info.level} events require start and end times")
                elif not (0 <= start < end <= info.duration_sec):
                    problems.append(
                        f"{video_id}: span ({start}, {end}) must satisfy "
                        f"0 <= start < end <= duration ({info.duration_sec})"
                    )
            explanation = event.get("explanation")
            if explanation is not None and not (
                EXPLANATION_MIN_CHARS <= len(explanation) <= EXPLANATION_MAX_CHARS
            ):
                problems.append(
                    f"{video_id}: explanation must be "
                    f"{EXPLANATION_MIN_CHARS}-{EXPLANATION_MAX_CHARS} characters"
                )

    for missing in manifest.keys() - seen:
        problems.append(f"{missing}: missing from predictions")

    return sorted(problems)


def write_submission(payload: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
