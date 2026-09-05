"""A single normalised event type, used for both ground truth and predictions."""

from dataclasses import dataclass

from ahc_vad.taxonomy import validate_anomaly_class

Span = tuple[float, float]


def temporal_iou(a: Span, b: Span) -> float:
    """Intersection-over-union of two [start, end] intervals in seconds.

    Returns 0.0 for disjoint, touching, or zero-length inputs.
    """
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    intersection = max(0.0, end - start)
    if intersection == 0.0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class Event:
    """One anomaly occurrence.

    `start_time_sec`/`end_time_sec` are both None for D1 (video-level) events, and both
    populated for D2/D3. Half-populated is always an error.
    """

    class_name: str
    start_time_sec: float | None
    end_time_sec: float | None
    explanation: str | None = None

    def __post_init__(self) -> None:
        validate_anomaly_class(self.class_name)
        has_start = self.start_time_sec is not None
        has_end = self.end_time_sec is not None
        if has_start != has_end:
            raise ValueError(
                "start_time_sec and end_time_sec must both be set or both be None; "
                f"got {self.start_time_sec!r} and {self.end_time_sec!r}"
            )
        if has_start and self.end_time_sec <= self.start_time_sec:
            raise ValueError(
                f"end_time_sec ({self.end_time_sec}) must be greater than "
                f"start_time_sec ({self.start_time_sec})"
            )

    @property
    def is_localised(self) -> bool:
        return self.start_time_sec is not None

    @property
    def span(self) -> Span:
        if not self.is_localised:
            raise ValueError(f"{self.class_name} event has no span (D1 events are unlocalised)")
        return (self.start_time_sec, self.end_time_sec)

    @property
    def duration(self) -> float:
        start, end = self.span
        return end - start
