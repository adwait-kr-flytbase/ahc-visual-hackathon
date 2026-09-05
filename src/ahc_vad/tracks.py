"""Per-object state layer for the DURATION-defined classes.

Rationale (measured by the inference lane, zero-shot Qwen3-VL-4B on the 24-video D1 set):
the model is confident on APPEARANCE classes and silent on CONTEXT/DURATION ones --
traffic_accident 3/3, smoke 2/2, waterlogging 2/2, but road_spill 0/2, fighting 0/2,
stalled 0/1, blocking 0/1, wrong_way 0/1. Five inference-time interventions (8/16/24/32
frames, two prompt variants) produced zero improvements.

`stalled_or_broken_down_vehicle`, `loitering_or_suspicious_presence`, `traffic_congestion`
and `vehicle_blocking_traffic` are not things you see in a frame -- they are things that
are true of an object over TIME. This derives them from tracks instead.

Camera motion is NOT compensated here: a global median-flow estimate is subtracted from
per-track velocity, which handles slow drone drift but not aggressive dashcam ego-motion.
Treat dashcam results as the weak case.
"""

from collections import defaultdict
from dataclasses import dataclass, field

# COCO ids from the default YOLO weights.
VEHICLE_IDS = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_ID = 0


@dataclass
class Track:
    track_id: int
    cls_id: int
    times: list[float] = field(default_factory=list)
    centres: list[tuple[float, float]] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    def speeds(self) -> list[tuple[float, float, float]]:
        """(time, dx_per_sec, dy_per_sec) between consecutive observations."""
        out = []
        for i in range(1, len(self.times)):
            dt = self.times[i] - self.times[i - 1]
            if dt <= 0:
                continue
            (x0, y0), (x1, y1) = self.centres[i - 1], self.centres[i]
            out.append((self.times[i], (x1 - x0) / dt, (y1 - y0) / dt))
        return out


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def ego_motion(tracks: list[Track], times: list[float]) -> dict[float, tuple[float, float]]:
    """Median velocity across all tracks at each timestamp -- a crude global-motion proxy.

    If the whole scene appears to move together, that is the camera, not the objects.
    """
    per_time: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for track in tracks:
        for t, dx, dy in track.speeds():
            per_time[round(t, 2)].append((dx, dy))
    return {
        t: (_median([v[0] for v in vs]), _median([v[1] for v in vs]))
        for t, vs in per_time.items()
    }


def _compensated_speed(track: Track, ego: dict[float, tuple[float, float]], diag: float):
    """Yield (time, speed) with global motion removed, normalised by the frame diagonal."""
    for t, dx, dy in track.speeds():
        ex, ey = ego.get(round(t, 2), (0.0, 0.0))
        rx, ry = dx - ex, dy - ey
        yield t, ((rx * rx + ry * ry) ** 0.5) / diag


def _runs(flags: list[tuple[float, bool]], min_len: float) -> list[tuple[float, float]]:
    """Contiguous spans where the flag holds, lasting at least `min_len` seconds."""
    spans, start, prev = [], None, None
    for t, ok in flags:
        if ok and start is None:
            start = t
        elif not ok and start is not None:
            if prev is not None and prev - start >= min_len:
                spans.append((start, prev))
            start = None
        prev = t
    if start is not None and prev is not None and prev - start >= min_len:
        spans.append((start, prev))
    return spans


def derive_events(
    tracks: list[Track],
    frame_times: list[float],
    width: int,
    height: int,
    *,
    stalled_min_sec: float = 8.0,
    loiter_min_sec: float = 12.0,
    congestion_min_sec: float = 6.0,
    still_speed: float = 0.004,
    congestion_min_vehicles: int = 8,
) -> list[dict]:
    """Turn tracks into candidate events for the four duration-defined classes."""
    if not frame_times:
        return []
    diag = (width**2 + height**2) ** 0.5
    ego = ego_motion(tracks, frame_times)
    events: list[dict] = []

    for track in tracks:
        samples = list(_compensated_speed(track, ego, diag))
        if len(samples) < 3:
            continue
        still = [(t, speed < still_speed) for t, speed in samples]

        if track.cls_id in VEHICLE_IDS:
            for start, end in _runs(still, stalled_min_sec):
                events.append({
                    "class_name": "stalled_or_broken_down_vehicle",
                    "start": round(start, 2), "end": round(end, 2),
                    "confidence": round(min(0.9, 0.4 + (end - start) / 60), 2),
                    "explanation": (
                        f"A {VEHICLE_IDS[track.cls_id]} remains stationary for "
                        f"{end - start:.0f} seconds while other traffic continues around it."
                    ),
                })
        elif track.cls_id == PERSON_ID:
            for start, end in _runs(still, loiter_min_sec):
                events.append({
                    "class_name": "loitering_or_suspicious_presence",
                    "start": round(start, 2), "end": round(end, 2),
                    "confidence": round(min(0.9, 0.4 + (end - start) / 60), 2),
                    "explanation": (
                        f"A person stays in the same location for {end - start:.0f} seconds "
                        f"without moving on."
                    ),
                })

    # Congestion: many vehicles present AND their median speed collapsing, sustained.
    per_time_speed: dict[float, list[float]] = defaultdict(list)
    per_time_count: dict[float, int] = defaultdict(int)
    for track in tracks:
        if track.cls_id not in VEHICLE_IDS:
            continue
        for t in track.times:
            per_time_count[round(t, 2)] += 1
        for t, speed in _compensated_speed(track, ego, diag):
            per_time_speed[round(t, 2)].append(speed)

    times = sorted(per_time_count)
    if times:
        moving = [_median(per_time_speed.get(t, [])) for t in times]
        reference = _median([m for m in moving if m > 0]) or 1e-6
        flags = [
            (t, per_time_count[t] >= congestion_min_vehicles and moving[i] < 0.45 * reference)
            for i, t in enumerate(times)
        ]
        for start, end in _runs(flags, congestion_min_sec):
            peak = max(per_time_count[t] for t in times if start <= t <= end)
            events.append({
                "class_name": "traffic_congestion",
                "start": round(start, 2), "end": round(end, 2),
                "confidence": 0.6,
                "explanation": (
                    f"Vehicle density rises to {peak} while median speed falls to under half "
                    f"the scene baseline, sustained for {end - start:.0f} seconds."
                ),
            })

    return merge_same_class(events)


def merge_same_class(events: list[dict], gap_tolerance: float = 2.0) -> list[dict]:
    """Union overlapping or near-touching events of the same class."""
    out: list[dict] = []
    for event in sorted(events, key=lambda e: (e["class_name"], e["start"])):
        if out and out[-1]["class_name"] == event["class_name"] and \
                event["start"] - out[-1]["end"] <= gap_tolerance:
            out[-1]["end"] = max(out[-1]["end"], event["end"])
            out[-1]["confidence"] = max(out[-1]["confidence"], event["confidence"])
        else:
            out.append(dict(event))
    return sorted(out, key=lambda e: e["start"])
