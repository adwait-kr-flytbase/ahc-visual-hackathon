import pytest
from ahc_vad.events import Event, temporal_iou


def test_identical_spans_have_iou_one():
    assert temporal_iou((10.0, 20.0), (10.0, 20.0)) == pytest.approx(1.0)


def test_disjoint_spans_have_iou_zero():
    assert temporal_iou((0.0, 10.0), (20.0, 30.0)) == 0.0


def test_touching_spans_have_iou_zero():
    assert temporal_iou((0.0, 10.0), (10.0, 20.0)) == 0.0


def test_half_overlap():
    assert temporal_iou((0.0, 20.0), (10.0, 30.0)) == pytest.approx(1 / 3)


def test_contained_span():
    assert temporal_iou((0.0, 20.0), (5.0, 10.0)) == pytest.approx(0.25)


def test_iou_is_symmetric():
    assert temporal_iou((3.0, 9.0), (5.0, 12.0)) == pytest.approx(temporal_iou((5.0, 12.0), (3.0, 9.0)))


def test_zero_length_spans_give_zero_not_a_crash():
    assert temporal_iou((5.0, 5.0), (5.0, 5.0)) == 0.0


def test_localised_event_exposes_span_and_duration():
    e = Event("traffic_accident", 170.0, 245.0)
    assert e.is_localised
    assert e.span == (170.0, 245.0)
    assert e.duration == pytest.approx(75.0)


def test_unlocalised_event_has_no_span():
    e = Event("fire", None, None)
    assert not e.is_localised
    with pytest.raises(ValueError):
        _ = e.span


def test_event_rejects_an_invalid_class():
    with pytest.raises(ValueError):
        Event("normal", None, None)


def test_event_rejects_end_before_start():
    with pytest.raises(ValueError):
        Event("fire", 10.0, 5.0)


def test_event_rejects_half_populated_span():
    with pytest.raises(ValueError):
        Event("fire", 10.0, None)
