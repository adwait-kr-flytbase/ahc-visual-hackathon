import json
from pathlib import Path

import pytest

from ahc_vad.events import Event
from ahc_vad.groundtruth import load_manifest
from ahc_vad.submission import (
    RuntimeMetadata, build_submission, validate_submission, write_submission,
)

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def manifest():
    return load_manifest(DATA / "manifest.json")


def test_empty_submission_is_valid_and_covers_every_video(manifest):
    payload = build_submission({}, manifest, submission_id="run-01", model_name="empty")
    assert validate_submission(payload, manifest) == []
    assert len(payload["predictions"]) == 34
    assert all(p["events"] == [] for p in payload["predictions"])


def test_d1_events_serialise_with_null_timestamps(manifest):
    payload = build_submission(
        {"T005": [Event("traffic_accident", None, None)]},
        manifest, submission_id="run-02", model_name="m",
    )
    event = next(p for p in payload["predictions"] if p["video_id"] == "T005")["events"][0]
    assert event["start_time_sec"] is None
    assert event["end_time_sec"] is None
    assert validate_submission(payload, manifest) == []


def test_d2_event_with_a_span_is_valid(manifest):
    payload = build_submission(
        {"T025": [Event("traffic_accident", 20.0, 40.0)]},
        manifest, submission_id="run-03", model_name="m",
    )
    assert validate_submission(payload, manifest) == []


def test_localised_event_at_d1_is_rejected(manifest):
    payload = build_submission(
        {"T005": [Event("traffic_accident", 1.0, 2.0)]},
        manifest, submission_id="run-04", model_name="m",
    )
    problems = validate_submission(payload, manifest)
    assert any("D1" in p and "T005" in p for p in problems)


def test_unlocalised_event_at_d2_is_rejected(manifest):
    payload = build_submission(
        {"T025": [Event("traffic_accident", None, None)]},
        manifest, submission_id="run-05", model_name="m",
    )
    assert any("T025" in p for p in validate_submission(payload, manifest))


def test_span_beyond_the_video_duration_is_rejected(manifest):
    payload = build_submission(
        {"T025": [Event("traffic_accident", 200.0, 300.0)]},
        manifest, submission_id="run-06", model_name="m",
    )
    assert any("duration" in p for p in validate_submission(payload, manifest))


def test_unknown_video_id_is_rejected(manifest):
    payload = build_submission({}, manifest, submission_id="run-07", model_name="m")
    payload["predictions"].append({"video_id": "T999", "events": [], "runtime_metadata": {}})
    assert any("T999" in p for p in validate_submission(payload, manifest))


def test_duplicate_video_id_is_rejected(manifest):
    payload = build_submission({}, manifest, submission_id="run-08", model_name="m")
    payload["predictions"].append(payload["predictions"][0])
    assert any("duplicate" in p.lower() for p in validate_submission(payload, manifest))


def test_short_explanation_is_rejected(manifest):
    payload = build_submission(
        {"T025": [Event("traffic_accident", 20.0, 40.0, explanation="too short")]},
        manifest, submission_id="run-09", model_name="m",
    )
    assert any("explanation" in p for p in validate_submission(payload, manifest))


def test_write_then_read_roundtrips(tmp_path, manifest):
    payload = build_submission({}, manifest, submission_id="run-10", model_name="m")
    out = tmp_path / "submission.json"
    write_submission(payload, out)
    assert json.loads(out.read_text()) == payload


def test_matches_the_portal_template_shape(manifest):
    template = json.loads((DATA / "submission-template.json").read_text())
    payload = build_submission({}, manifest, submission_id="x", model_name="y")
    assert payload.keys() == template.keys()
    assert payload["predictions"][0].keys() == template["predictions"][0].keys()
    assert (
        payload["predictions"][0]["runtime_metadata"].keys()
        == template["predictions"][0]["runtime_metadata"].keys()
    )
