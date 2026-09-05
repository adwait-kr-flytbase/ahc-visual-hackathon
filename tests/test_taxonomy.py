import pytest
from ahc_vad.taxonomy import (
    ANOMALY_CLASSES, NORMAL_CLASS, is_valid_anomaly_class, validate_anomaly_class,
)


def test_exactly_eleven_anomaly_classes():
    assert len(ANOMALY_CLASSES) == 11


def test_normal_is_not_an_anomaly_class():
    assert NORMAL_CLASS == "normal"
    assert NORMAL_CLASS not in ANOMALY_CLASSES
    assert not is_valid_anomaly_class("normal")


def test_known_classes_are_valid():
    for name in ("traffic_accident", "loitering_or_suspicious_presence", "wrong_way_driving"):
        assert is_valid_anomaly_class(name)


def test_validate_returns_the_name_when_valid():
    assert validate_anomaly_class("fire") == "fire"


@pytest.mark.parametrize("bad", ["Fire", "fire ", "traffic accident", "", "normal", "unknown"])
def test_validate_rejects_invalid_names(bad):
    with pytest.raises(ValueError):
        validate_anomaly_class(bad)
