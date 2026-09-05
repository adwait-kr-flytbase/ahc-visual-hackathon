"""The submission taxonomy.

11 anomaly classes, not 12: the dataset ships a `normal/` training folder, but `normal`
is not submittable -- absence of anomaly is an empty `events` list.
See .context/07-platform-and-scoring.md.
"""

NORMAL_CLASS = "normal"

ANOMALY_CLASSES = frozenset({
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "wrong_way_driving",
    "road_spill_or_debris",
    "waterlogging_or_flood",
    "fire",
    "smoke",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
})


def is_valid_anomaly_class(name: str) -> bool:
    """True if `name` is one of the 11 submittable class strings, matched exactly."""
    return name in ANOMALY_CLASSES


def validate_anomaly_class(name: str) -> str:
    """Return `name` unchanged, or raise ValueError. Exact match -- no casefolding, no stripping."""
    if not is_valid_anomaly_class(name):
        raise ValueError(
            f"{name!r} is not a submittable class. Expected one of: {sorted(ANOMALY_CLASSES)}"
        )
    return name
