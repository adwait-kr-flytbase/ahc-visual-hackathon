# Step 1 — Scoring Harness & Submission Emitter

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement task-by-task. Steps use `- [ ]`
> checkboxes for tracking.

**Status:** `READY` — implement as written.

**Goal:** Build a local, offline event scorer and a schema-valid submission writer, so every later
step can be measured in seconds without uploading, and so the first upload can happen today.

**Architecture:** Five small pure-Python modules under `src/ahc_vad/`, no ML dependencies. Ground
truth and predictions are normalised into one `Event` type; a greedy matcher pairs them under a
configurable policy; an aggregator reports precision / recall / false alarms per difficulty and per
class. A script emits the all-empty submission.

**Tech Stack:** Python 3.11+, stdlib only (`csv`, `json`, `dataclasses`, `argparse`), pytest for tests.
**No numpy, no pandas** — the data is 52 ground-truth rows and 34 videos; stdlib is faster to install
and impossible to get wrong.

**Spec context:** [`../.context/07-platform-and-scoring.md`](../.context/07-platform-and-scoring.md)
(submission schema, marks), [`../.context/09-dataset-profile.md`](../.context/09-dataset-profile.md)
(measured data facts).

---

## Global Constraints

- **Taxonomy is exactly 11 anomaly classes.** `normal` is NEVER emitted; it is `"events": []`.
- **Difficulty tiers:** D1 = `level 1` (24 videos), D2 = `level 2` (6), D3 = `level 3` (4).
  Levels come from `data/manifest.json` only.
- **D1 predictions must have `start_time_sec` and `end_time_sec` equal to `null`.**
  D2/D3 predictions must have both populated, `0 <= start < end <= duration_sec`.
- **All times are floats in seconds.**
- **Determinism:** every ordering has an explicit stable tie-break.
- `explanation` is optional; when present it must be 20–500 characters.
- Every one of the 34 manifest video ids appears exactly once in `predictions`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | pytest config; puts `src/` on the import path. No packaging. |
| `data/manifest.json` | Canonical copy of the portal manifest (id → level, duration). |
| `data/submission-template.json` | Canonical copy of the portal template. |
| `src/ahc_vad/taxonomy.py` | The 11 class strings + validation. Single source of truth. |
| `src/ahc_vad/events.py` | `Event` dataclass, `temporal_iou`. |
| `src/ahc_vad/groundtruth.py` | Load `manifest.json` and a `ground_truth.csv` into `Event`s. |
| `src/ahc_vad/submission.py` | Build, validate and write submission JSON. |
| `src/ahc_vad/scoring.py` | `MatchPolicy`, greedy matcher, aggregation, report. |
| `scripts/make_empty_submission.py` | Emit the all-empty submission file. |
| `scripts/score_submission.py` | CLI: score a submission file against ground truth. |
| `tests/…` | One test module per source module. |

---

## Known unknown — resolve during this step

`.context/07-platform-and-scoring.md` records **both** a *15 s boundary tolerance* and
*tIoU ≥ 0.5*. These are different rules and it is unclear how they combine. On a 5 s event (7 of the
18 D2 events) 15 s tolerance is trivially met while tIoU ≥ 0.5 is strict.

**Therefore `MatchPolicy` is configurable and defaults to strict tIoU ≥ 0.5**, with
`boundary_tolerance_sec` available as an alternative acceptance path. Task 7 probes the portal to
settle it. Do not hard-code either rule.

---

## Task 1: Project skeleton and taxonomy

**Files:**
- Create: `pyproject.toml`, `src/ahc_vad/__init__.py`, `src/ahc_vad/taxonomy.py`
- Create: `tests/test_taxonomy.py`
- Copy: `.context/artifacts/manifest.json` → `data/manifest.json`;
  `.context/artifacts/submission-template.json` → `data/submission-template.json`

**Interfaces:**
- Produces: `ANOMALY_CLASSES: frozenset[str]`, `NORMAL_CLASS: str`,
  `is_valid_anomaly_class(name: str) -> bool`, `validate_anomaly_class(name: str) -> str`

- [ ] **Step 1: Initialise the repo and directories**

The repo is already initialised and `.gitignore` is already written — **do not overwrite it**.

```bash
mkdir -p src/ahc_vad tests scripts data
cp .context/artifacts/manifest.json data/manifest.json
cp .context/artifacts/submission-template.json data/submission-template.json
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = ["integration: touches the downloaded dataset/ pack"]
```

- [ ] **Step 3: Write the failing test**

`tests/test_taxonomy.py`:
```python
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
```

- [ ] **Step 4: Run the test and confirm it fails**

Run: `pytest tests/test_taxonomy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ahc_vad.taxonomy'`

- [ ] **Step 5: Implement `src/ahc_vad/taxonomy.py`**

```python
"""The submission taxonomy.

11 anomaly classes, not 12: the dataset ships a `normal/` training folder, but `normal`
is not submittable — absence of anomaly is an empty `events` list.
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
    """Return `name` unchanged, or raise ValueError. Exact match — no casefolding, no stripping."""
    if not is_valid_anomaly_class(name):
        raise ValueError(
            f"{name!r} is not a submittable class. "
            f"Expected one of: {sorted(ANOMALY_CLASSES)}"
        )
    return name
```

Also create an empty `src/ahc_vad/__init__.py`.

- [ ] **Step 6: Run the test and confirm it passes**

Run: `pytest tests/test_taxonomy.py -v`
Expected: PASS, 8 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore data src tests
git commit -m "feat: project skeleton and 11-class submission taxonomy"
```

---

## Task 2: Event type and temporal IoU

**Files:**
- Create: `src/ahc_vad/events.py`, `tests/test_events.py`

**Interfaces:**
- Consumes: `ahc_vad.taxonomy.validate_anomaly_class`
- Produces:
  - `Event(class_name: str, start_time_sec: float | None, end_time_sec: float | None, explanation: str | None = None)`
  - `Event.is_localised -> bool`, `Event.span -> tuple[float, float]`, `Event.duration -> float`
  - `temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float`

- [ ] **Step 1: Write the failing test**

`tests/test_events.py`:
```python
import pytest
from ahc_vad.events import Event, temporal_iou


def test_identical_spans_have_iou_one():
    assert temporal_iou((10.0, 20.0), (10.0, 20.0)) == pytest.approx(1.0)


def test_disjoint_spans_have_iou_zero():
    assert temporal_iou((0.0, 10.0), (20.0, 30.0)) == 0.0


def test_touching_spans_have_iou_zero():
    assert temporal_iou((0.0, 10.0), (10.0, 20.0)) == 0.0


def test_half_overlap():
    # inter = 10, union = 30 -> 1/3
    assert temporal_iou((0.0, 20.0), (10.0, 30.0)) == pytest.approx(1 / 3)


def test_contained_span():
    # inter = 5, union = 20 -> 0.25
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ahc_vad.events'`

- [ ] **Step 3: Implement `src/ahc_vad/events.py`**

```python
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
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pytest tests/test_events.py -v`
Expected: PASS, 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/ahc_vad/events.py tests/test_events.py
git commit -m "feat: Event type and temporal IoU"
```

---

## Task 3: Load the manifest and ground truth

**Files:**
- Create: `src/ahc_vad/groundtruth.py`, `tests/test_groundtruth.py`

**Interfaces:**
- Consumes: `ahc_vad.events.Event`
- Produces:
  - `VideoInfo(video_id: str, level: int, duration_sec: float)`
  - `load_manifest(path: str | Path) -> dict[str, VideoInfo]`
  - `load_ground_truth(path: str | Path) -> dict[str, list[Event]]`

**Key behaviour:** a CSV row with `class_name == "normal"` means *this video has no events*. The
video must still appear in the returned mapping, with an empty list. That is what makes T029/T030
(240 s of normal footage at D2) scoreable as false-alarm traps.

- [ ] **Step 1: Write the failing test**

`tests/test_groundtruth.py`:
```python
import textwrap
from pathlib import Path

import pytest

from ahc_vad.groundtruth import load_ground_truth, load_manifest

DATA = Path(__file__).resolve().parents[1] / "data"
DATASET = Path(__file__).resolve().parents[1] / "dataset"


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "gt.csv"
    p.write_text(textwrap.dedent(body).lstrip())
    return p


def test_normal_row_yields_a_video_with_no_events(tmp_path):
    path = _write(tmp_path, """
        video_id,level,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary
        T029,2,false,normal,,,
    """)
    gt = load_ground_truth(path)
    assert gt == {"T029": []}


def test_multiple_rows_for_one_video_become_multiple_events(tmp_path):
    path = _write(tmp_path, """
        video_id,level,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary
        T033,3,true,traffic_accident,170,245,First
        T033,3,true,traffic_accident,490,535,Second
    """)
    gt = load_ground_truth(path)
    assert len(gt["T033"]) == 2
    assert gt["T033"][0].span == (170.0, 245.0)
    assert gt["T033"][1].span == (490.0, 535.0)


def test_blank_timestamps_produce_an_unlocalised_event(tmp_path):
    path = _write(tmp_path, """
        video_id,level,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary
        T005,1,true,traffic_accident,,,Crashed cars
    """)
    gt = load_ground_truth(path)
    assert not gt["T005"][0].is_localised


def test_commas_inside_the_description_do_not_break_parsing(tmp_path):
    path = _write(tmp_path, '''
        video_id,level,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary
        T001,1,true,fire,1,2,"A fire, with smoke, spreads"
    ''')
    gt = load_ground_truth(path)
    assert gt["T001"][0].explanation == "A fire, with smoke, spreads"


def test_train_csv_without_a_level_column_still_loads(tmp_path):
    # train/*/ground_truth.csv has no `level` column at all.
    path = _write(tmp_path, """
        video_id,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary
        TR01350,true,traffic_accident,0.000,5.000,A traffic collision occurs.
    """)
    gt = load_ground_truth(path)
    assert gt["TR01350"][0].span == (0.0, 5.0)


def test_manifest_loads_levels_and_durations():
    manifest = load_manifest(DATA / "manifest.json")
    assert len(manifest) == 34
    assert manifest["T001"].level == 1
    assert manifest["T025"].level == 2
    assert manifest["T025"].duration_sec == pytest.approx(240.0)
    assert manifest["T034"].level == 3


@pytest.mark.integration
def test_public_ground_truth_matches_the_measured_profile():
    gt_path = DATASET / "test" / "ground_truth.csv"
    if not gt_path.exists():
        pytest.skip("dataset pack not present")
    gt = load_ground_truth(gt_path)
    manifest = load_manifest(DATA / "manifest.json")
    assert set(gt) == set(manifest)
    by_level = {1: 0, 2: 0, 3: 0}
    for vid, events in gt.items():
        by_level[manifest[vid].level] += len(events)
    # Measured in .context/09-dataset-profile.md and confirmed by the leaderboard denominators.
    assert by_level == {1: 20, 2: 18, 3: 8}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_groundtruth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ahc_vad.groundtruth'`

- [ ] **Step 3: Implement `src/ahc_vad/groundtruth.py`**

```python
"""Readers for the portal manifest and the dataset ground-truth CSVs.

Two shapes exist:
  - test/ground_truth.csv     has a `level` column
  - train/<class>/ground_truth.csv  does NOT
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
    in the mapping with an empty list — normal videos are scoreable false-alarm traps.
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
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pytest tests/test_groundtruth.py -v`
Expected: PASS, 7 passed (the last is skipped if `dataset/` is absent)

- [ ] **Step 5: Commit**

```bash
git add src/ahc_vad/groundtruth.py tests/test_groundtruth.py
git commit -m "feat: manifest and ground-truth loaders"
```

---

## Task 4: Submission builder, validator and the first upload

**Files:**
- Create: `src/ahc_vad/submission.py`, `tests/test_submission.py`, `scripts/make_empty_submission.py`

**Interfaces:**
- Consumes: `Event`, `VideoInfo`, `load_manifest`
- Produces:
  - `RuntimeMetadata(frames_processed: int = 0, chunks_processed: int = 1, end_to_end_internal_time_ms: float = 0.0, model_runtimes: list[dict] | None = None)`
  - `build_submission(predictions: dict[str, list[Event]], manifest: dict[str, VideoInfo], *, submission_id: str, model_name: str, runtimes: dict[str, RuntimeMetadata] | None = None, hardware: str = "unspecified", total_wall_time_ms: float = 0.0) -> dict`
  - `validate_submission(payload: dict, manifest: dict[str, VideoInfo]) -> list[str]` — returns
    human-readable problems; empty list means valid
  - `write_submission(payload: dict, path: str | Path) -> None`

**This task delivers the first real upload.** An all-empty submission is worth marks on its own
(≈13.5/100 by leaderboard inference) and proves the schema end to end before any model exists.

- [ ] **Step 1: Write the failing test**

`tests/test_submission.py`:
```python
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
    problems = validate_submission(payload, manifest)
    assert any("T025" in p for p in problems)


def test_span_beyond_the_video_duration_is_rejected(manifest):
    payload = build_submission(
        {"T025": [Event("traffic_accident", 200.0, 300.0)]},  # duration is 240 s
        manifest, submission_id="run-06", model_name="m",
    )
    problems = validate_submission(payload, manifest)
    assert any("duration" in p for p in problems)


def test_unknown_video_id_is_rejected(manifest):
    payload = build_submission({}, manifest, submission_id="run-07", model_name="m")
    payload["predictions"].append({"video_id": "T999", "events": [], "runtime_metadata": {}})
    problems = validate_submission(payload, manifest)
    assert any("T999" in p for p in problems)


def test_duplicate_video_id_is_rejected(manifest):
    payload = build_submission({}, manifest, submission_id="run-08", model_name="m")
    payload["predictions"].append(payload["predictions"][0])
    problems = validate_submission(payload, manifest)
    assert any("duplicate" in p.lower() for p in problems)


def test_short_explanation_is_rejected(manifest):
    payload = build_submission(
        {"T025": [Event("traffic_accident", 20.0, 40.0, explanation="too short")]},
        manifest, submission_id="run-09", model_name="m",
    )
    problems = validate_submission(payload, manifest)
    assert any("explanation" in p for p in problems)


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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_submission.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ahc_vad.submission'`

- [ ] **Step 3: Implement `src/ahc_vad/submission.py`**

```python
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
    """Self-reported per-video timings. Feeds the latency bonus — report honestly."""

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
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pytest tests/test_submission.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Write `scripts/make_empty_submission.py`**

```python
"""Emit an all-empty submission: every video predicted normal.

This is the correct first upload — it validates the schema end to end and, per the
leaderboard, scores a non-zero floor on its own.

    python scripts/make_empty_submission.py --out out/submission-empty.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ahc_vad.groundtruth import load_manifest
from ahc_vad.submission import build_submission, validate_submission, write_submission

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=ROOT / "data" / "manifest.json", type=Path)
    parser.add_argument("--out", default=ROOT / "out" / "submission-empty.json", type=Path)
    parser.add_argument("--submission-id", default="empty-baseline-01")
    parser.add_argument("--model-name", default="all-normal-baseline")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    payload = build_submission(
        {}, manifest, submission_id=args.submission_id, model_name=args.model_name
    )
    problems = validate_submission(payload, manifest)
    if problems:
        print("INVALID submission:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    write_submission(payload, args.out)
    print(f"Wrote {args.out} — {len(payload['predictions'])} videos, 0 events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Generate the file and eyeball it**

Run: `python scripts/make_empty_submission.py`
Expected: `Wrote out/submission-empty.json — 34 videos, 0 events.`

- [ ] **Step 7: Commit**

```bash
git add src/ahc_vad/submission.py tests/test_submission.py scripts/make_empty_submission.py
git commit -m "feat: submission builder, validator and all-empty baseline"
```

---

## Task 5: The event matcher

**Files:**
- Create: `src/ahc_vad/scoring.py`, `tests/test_scoring.py`

**Interfaces:**
- Consumes: `Event`, `temporal_iou`
- Produces:
  - `MatchPolicy(iou_threshold: float = 0.5, boundary_tolerance_sec: float | None = None, require_temporal: bool = True)`
  - `MatchResult(matched: list[tuple[int, int, float]], unmatched_gt: list[int], unmatched_pred: list[int])`
  - `match_events(gt: list[Event], pred: list[Event], policy: MatchPolicy) -> MatchResult`

**Matching rule.** A ground-truth event and a predicted event may pair only if their
`class_name` strings are equal. When `require_temporal` is true they must also satisfy the policy:
tIoU ≥ `iou_threshold`, **or** — if `boundary_tolerance_sec` is set — both endpoints within that
tolerance. Candidate pairs are sorted by IoU descending, then by ground-truth index, then by
prediction index, and consumed greedily one-to-one. The tie-break makes the result deterministic.

- [ ] **Step 1: Write the failing test**

`tests/test_scoring.py`:
```python
import pytest

from ahc_vad.events import Event
from ahc_vad.scoring import MatchPolicy, match_events

D1 = MatchPolicy(require_temporal=False)
D23 = MatchPolicy(iou_threshold=0.5)


def test_d1_matches_on_class_alone():
    gt = [Event("fire", None, None)]
    pred = [Event("fire", None, None)]
    result = match_events(gt, pred, D1)
    assert len(result.matched) == 1
    assert result.unmatched_gt == [] and result.unmatched_pred == []


def test_d1_wrong_class_is_one_miss_and_one_false_alarm():
    gt = [Event("fire", None, None)]
    pred = [Event("smoke", None, None)]
    result = match_events(gt, pred, D1)
    assert result.matched == []
    assert result.unmatched_gt == [0] and result.unmatched_pred == [0]


def test_exact_span_match():
    gt = [Event("traffic_accident", 20.0, 40.0)]
    pred = [Event("traffic_accident", 20.0, 40.0)]
    assert len(match_events(gt, pred, D23).matched) == 1


def test_span_below_the_iou_threshold_does_not_match():
    # inter = 5, union = 35 -> 0.143
    gt = [Event("traffic_accident", 20.0, 40.0)]
    pred = [Event("traffic_accident", 35.0, 55.0)]
    result = match_events(gt, pred, D23)
    assert result.matched == []
    assert result.unmatched_pred == [0]


def test_span_just_above_the_iou_threshold_matches():
    # gt 20-40, pred 25-45: inter = 15, union = 25 -> 0.6
    gt = [Event("traffic_accident", 20.0, 40.0)]
    pred = [Event("traffic_accident", 25.0, 45.0)]
    assert len(match_events(gt, pred, D23).matched) == 1


def test_right_span_but_wrong_class_never_matches():
    gt = [Event("traffic_accident", 20.0, 40.0)]
    pred = [Event("fire", 20.0, 40.0)]
    assert match_events(gt, pred, D23).matched == []


def test_one_prediction_cannot_claim_two_ground_truth_events():
    gt = [Event("traffic_accident", 20.0, 40.0), Event("traffic_accident", 60.0, 80.0)]
    pred = [Event("traffic_accident", 20.0, 40.0)]
    result = match_events(gt, pred, D23)
    assert len(result.matched) == 1
    assert result.unmatched_gt == [1]


def test_greedy_matching_prefers_the_higher_iou_pair():
    gt = [Event("traffic_accident", 20.0, 40.0)]
    pred = [
        Event("traffic_accident", 22.0, 44.0),  # iou ~0.75
        Event("traffic_accident", 20.0, 40.0),  # iou 1.0 -> must win
    ]
    result = match_events(gt, pred, D23)
    assert [pi for _, pi, _ in result.matched] == [1]
    assert result.unmatched_pred == [0]


def test_boundary_tolerance_accepts_what_iou_rejects():
    # A 5 s event predicted 6 s late: iou = 0, but both endpoints within 15 s.
    gt = [Event("traffic_accident", 30.0, 35.0)]
    pred = [Event("traffic_accident", 41.0, 46.0)]
    assert match_events(gt, pred, MatchPolicy(iou_threshold=0.5)).matched == []
    lenient = MatchPolicy(iou_threshold=0.5, boundary_tolerance_sec=15.0)
    assert len(match_events(gt, pred, lenient).matched) == 1


def test_empty_prediction_against_normal_video_is_clean():
    result = match_events([], [], D23)
    assert result.matched == [] and result.unmatched_gt == [] and result.unmatched_pred == []


def test_prediction_on_a_normal_video_is_a_false_alarm():
    result = match_events([], [Event("fire", 1.0, 2.0)], D23)
    assert result.unmatched_pred == [0]


def test_matching_is_deterministic_across_repeated_calls():
    gt = [Event("traffic_accident", 20.0, 40.0), Event("traffic_accident", 21.0, 41.0)]
    pred = [Event("traffic_accident", 20.5, 40.5), Event("traffic_accident", 20.6, 40.6)]
    first = match_events(gt, pred, D23)
    for _ in range(20):
        assert match_events(gt, pred, D23) == first
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ahc_vad.scoring'`

- [ ] **Step 3: Implement the matcher half of `src/ahc_vad/scoring.py`**

```python
"""Local event matching and scoring.

This is a *matcher*, not a replica of the portal's marks formula — the floor component of
25/35/40 is not yet reverse-engineered. For iteration only the ranking has to be faithful.
See .context/07-platform-and-scoring.md.
"""

from dataclasses import dataclass

from ahc_vad.events import Event, temporal_iou


@dataclass(frozen=True)
class MatchPolicy:
    """How a predicted event is allowed to satisfy a ground-truth event.

    iou_threshold        minimum temporal IoU (the documented D2/D3 rule)
    boundary_tolerance_sec  if set, an ALTERNATIVE acceptance path: both endpoints within
                         this many seconds. The portal documents a 15 s tolerance as well as
                         IoU >= 0.5 and it is unclear how they combine — keep this
                         configurable until a probe upload settles it.
    require_temporal     False for D1, where only the class is scored.
    """

    iou_threshold: float = 0.5
    boundary_tolerance_sec: float | None = None
    require_temporal: bool = True


@dataclass(frozen=True)
class MatchResult:
    matched: list[tuple[int, int, float]]  # (gt index, pred index, iou)
    unmatched_gt: list[int]
    unmatched_pred: list[int]

    @property
    def true_positives(self) -> int:
        return len(self.matched)

    @property
    def false_alarms(self) -> int:
        return len(self.unmatched_pred)

    @property
    def misses(self) -> int:
        return len(self.unmatched_gt)


def _candidate_iou(gt: Event, pred: Event, policy: MatchPolicy) -> float | None:
    """IoU if this pair is allowed to match, else None."""
    if gt.class_name != pred.class_name:
        return None
    if not policy.require_temporal:
        return 1.0
    if not (gt.is_localised and pred.is_localised):
        return None
    iou = temporal_iou(gt.span, pred.span)
    if iou >= policy.iou_threshold:
        return iou
    if policy.boundary_tolerance_sec is not None:
        tolerance = policy.boundary_tolerance_sec
        if (
            abs(gt.start_time_sec - pred.start_time_sec) <= tolerance
            and abs(gt.end_time_sec - pred.end_time_sec) <= tolerance
        ):
            return iou
    return None


def match_events(gt: list[Event], pred: list[Event], policy: MatchPolicy) -> MatchResult:
    """Greedy one-to-one matching, highest IoU first.

    Ties break on ground-truth index then prediction index, so the result is deterministic.
    """
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_event in enumerate(gt):
        for pred_index, pred_event in enumerate(pred):
            iou = _candidate_iou(gt_event, pred_event, policy)
            if iou is not None:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matched.append((gt_index, pred_index, iou))

    matched.sort(key=lambda item: item[0])
    return MatchResult(
        matched=matched,
        unmatched_gt=[i for i in range(len(gt)) if i not in used_gt],
        unmatched_pred=[i for i in range(len(pred)) if i not in used_pred],
    )
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS, 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/ahc_vad/scoring.py tests/test_scoring.py
git commit -m "feat: deterministic greedy event matcher with configurable policy"
```

---

## Task 6: Aggregate into a per-difficulty report

**Files:**
- Modify: `src/ahc_vad/scoring.py` (append)
- Modify: `tests/test_scoring.py` (append)
- Create: `scripts/score_submission.py`

**Interfaces:**
- Produces:
  - `DifficultyScore(level: int, true_positives: int, false_alarms: int, misses: int, precision: float, recall: float, f1: float)`
  - `ScoreReport(by_difficulty: dict[int, DifficultyScore], by_class: dict[str, DifficultyScore], proxy_score: float)`
  - `score(gt: dict[str, list[Event]], pred: dict[str, list[Event]], manifest: dict[str, VideoInfo], *, policy_d1: MatchPolicy | None = None, policy_d23: MatchPolicy | None = None) -> ScoreReport`
  - `format_report(report: ScoreReport) -> str`

**`proxy_score` is `25*f1(D1) + 35*f1(D2) + 40*f1(D3)`.** It is explicitly **not** the portal's
marks — the floor component is unknown. It exists so runs can be ranked against each other.

- [ ] **Step 1: Append the failing tests to `tests/test_scoring.py`**

```python
from pathlib import Path

from ahc_vad.groundtruth import VideoInfo, load_ground_truth, load_manifest
from ahc_vad.scoring import format_report, score

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


def _manifest():
    return {
        "V1": VideoInfo("V1", 1, 10.0),
        "V2": VideoInfo("V2", 2, 240.0),
        "V3": VideoInfo("V3", 3, 300.0),
    }


def test_perfect_prediction_scores_full_proxy():
    gt = {
        "V1": [Event("fire", None, None)],
        "V2": [Event("traffic_accident", 20.0, 40.0)],
        "V3": [Event("traffic_congestion", 100.0, 200.0)],
    }
    report = score(gt, gt, _manifest())
    assert report.proxy_score == pytest.approx(100.0)
    assert report.by_difficulty[1].precision == 1.0
    assert report.by_difficulty[3].recall == 1.0


def test_empty_prediction_scores_zero_but_logs_no_false_alarms():
    gt = {
        "V1": [Event("fire", None, None)],
        "V2": [Event("traffic_accident", 20.0, 40.0)],
        "V3": [Event("traffic_congestion", 100.0, 200.0)],
    }
    report = score(gt, {}, _manifest())
    assert report.proxy_score == 0.0
    assert all(d.false_alarms == 0 for d in report.by_difficulty.values())
    assert report.by_difficulty[1].misses == 1


def test_false_alarms_on_a_normal_video_are_counted():
    gt = {"V1": [], "V2": [], "V3": []}
    pred = {"V2": [Event("fire", 10.0, 20.0), Event("smoke", 30.0, 40.0)]}
    report = score(gt, pred, _manifest())
    assert report.by_difficulty[2].false_alarms == 2
    assert report.by_difficulty[2].precision == 0.0


def test_per_class_breakdown_is_reported():
    gt = {"V2": [Event("fire", 10.0, 20.0), Event("smoke", 30.0, 40.0)]}
    pred = {"V2": [Event("fire", 10.0, 20.0)]}
    report = score(gt, pred, _manifest())
    assert report.by_class["fire"].true_positives == 1
    assert report.by_class["smoke"].misses == 1


def test_d1_ignores_timestamps_entirely():
    gt = {"V1": [Event("fire", None, None)]}
    pred = {"V1": [Event("fire", None, None)]}
    assert score(gt, pred, _manifest()).by_difficulty[1].f1 == 1.0


def test_format_report_mentions_each_difficulty():
    text = format_report(score({"V1": []}, {}, _manifest()))
    for label in ("D1", "D2", "D3"):
        assert label in text


@pytest.mark.integration
def test_public_ground_truth_scored_against_itself_is_perfect():
    gt_path = DATASET_DIR / "test" / "ground_truth.csv"
    if not gt_path.exists():
        pytest.skip("dataset pack not present")
    gt = load_ground_truth(gt_path)
    manifest = load_manifest(DATA_DIR / "manifest.json")
    report = score(gt, gt, manifest)
    assert report.proxy_score == pytest.approx(100.0)
    assert report.by_difficulty[1].true_positives == 20
    assert report.by_difficulty[2].true_positives == 18
    assert report.by_difficulty[3].true_positives == 8
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_scoring.py -v -k "report or perfect or false_alarms or per_class or d1_ignores"`
Expected: FAIL — `ImportError: cannot import name 'score' from 'ahc_vad.scoring'`

- [ ] **Step 3: Append the aggregator to `src/ahc_vad/scoring.py`**

```python
from ahc_vad.groundtruth import VideoInfo

DIFFICULTY_MARKS = {1: 25.0, 2: 35.0, 3: 40.0}


@dataclass(frozen=True)
class DifficultyScore:
    level: int
    true_positives: int
    false_alarms: int
    misses: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_alarms
        return self.true_positives / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.misses
        return self.true_positives / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass(frozen=True)
class ScoreReport:
    by_difficulty: dict[int, DifficultyScore]
    by_class: dict[str, DifficultyScore]
    proxy_score: float


def _tally(counts: dict, key, result: MatchResult) -> None:
    """Accumulate [true_positives, false_alarms, misses] under `key`."""
    entry = counts.setdefault(key, [0, 0, 0])
    entry[0] += result.true_positives
    entry[1] += result.false_alarms
    entry[2] += result.misses


def score(
    gt: dict[str, list[Event]],
    pred: dict[str, list[Event]],
    manifest: dict[str, VideoInfo],
    *,
    policy_d1: MatchPolicy | None = None,
    policy_d23: MatchPolicy | None = None,
) -> ScoreReport:
    """Score predictions against ground truth, split by difficulty and by class.

    Videos in the manifest but absent from `pred` are treated as predicting nothing.
    """
    policy_d1 = policy_d1 or MatchPolicy(require_temporal=False)
    policy_d23 = policy_d23 or MatchPolicy()

    per_level: dict[int, list[int]] = {}
    per_class: dict[str, list[int]] = {}

    for video_id, info in manifest.items():
        gt_events = gt.get(video_id, [])
        pred_events = pred.get(video_id, [])
        policy = policy_d1 if info.level == 1 else policy_d23
        result = match_events(gt_events, pred_events, policy)
        _tally(per_level, info.level, result)

        matched_gt = {gi for gi, _, _ in result.matched}
        matched_pred = {pi for _, pi, _ in result.matched}
        for index, event in enumerate(gt_events):
            entry = per_class.setdefault(event.class_name, [0, 0, 0])
            if index in matched_gt:
                entry[0] += 1
            else:
                entry[2] += 1
        for index, event in enumerate(pred_events):
            if index not in matched_pred:
                per_class.setdefault(event.class_name, [0, 0, 0])[1] += 1

    by_difficulty = {
        level: DifficultyScore(level, *per_level.get(level, [0, 0, 0]))
        for level in sorted(DIFFICULTY_MARKS)
    }
    # Per-class counts span all difficulties, so `level` is meaningless here; 0 marks it unused.
    by_class = {
        name: DifficultyScore(0, *counts) for name, counts in sorted(per_class.items())
    }
    proxy = sum(DIFFICULTY_MARKS[level] * by_difficulty[level].f1 for level in DIFFICULTY_MARKS)
    return ScoreReport(by_difficulty=by_difficulty, by_class=by_class, proxy_score=proxy)


def format_report(report: ScoreReport) -> str:
    lines = [
        f"proxy score {report.proxy_score:6.2f} / 100   "
        "(NOT the portal's marks - ranking aid only)",
        "",
        f"{'':<6}{'marks':>7}{'P':>7}{'R':>7}{'F1':>7}{'found':>7}{'FA':>5}{'miss':>6}",
    ]
    for level, entry in report.by_difficulty.items():
        lines.append(
            f"D{level:<5}{DIFFICULTY_MARKS[level]:>7.0f}{entry.precision:>7.2f}"
            f"{entry.recall:>7.2f}{entry.f1:>7.2f}{entry.true_positives:>7}"
            f"{entry.false_alarms:>5}{entry.misses:>6}"
        )
    lines += ["", f"{'class':<34}{'found':>7}{'FA':>5}{'miss':>6}"]
    for name, entry in report.by_class.items():
        lines.append(
            f"{name:<34}{entry.true_positives:>7}{entry.false_alarms:>5}{entry.misses:>6}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run the whole suite and confirm it passes**

Run: `pytest -v`
Expected: PASS — all tests green

- [ ] **Step 5: Write `scripts/score_submission.py`**

```python
"""Score a submission file against a ground-truth CSV.

    python scripts/score_submission.py out/submission-empty.json
    python scripts/score_submission.py out/run.json --boundary-tolerance 15
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ahc_vad.events import Event
from ahc_vad.groundtruth import load_ground_truth, load_manifest
from ahc_vad.scoring import MatchPolicy, format_report, score

ROOT = Path(__file__).resolve().parents[1]


def load_predictions(path: Path) -> dict[str, list[Event]]:
    payload = json.loads(path.read_text())
    return {
        prediction["video_id"]: [
            Event(
                class_name=event["class_name"],
                start_time_sec=event.get("start_time_sec"),
                end_time_sec=event.get("end_time_sec"),
                explanation=event.get("explanation"),
            )
            for event in prediction["events"]
        ]
        for prediction in payload["predictions"]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--ground-truth", type=Path,
                        default=ROOT / "dataset" / "test" / "ground_truth.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.json")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance", type=float, default=None,
                        help="seconds; enables the alternative acceptance path")
    args = parser.parse_args()

    report = score(
        load_ground_truth(args.ground_truth),
        load_predictions(args.submission),
        load_manifest(args.manifest),
        policy_d23=MatchPolicy(
            iou_threshold=args.iou, boundary_tolerance_sec=args.boundary_tolerance
        ),
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Score the empty submission and record the number**

Run: `python scripts/score_submission.py out/submission-empty.json`
Expected: proxy score `0.00`, **all false-alarm counts zero**, misses 20 / 18 / 8.
That zero-FA result is the point: it is the clean floor every later run must beat on precision.

- [ ] **Step 7: Commit**

```bash
git add src/ahc_vad/scoring.py tests/test_scoring.py scripts/score_submission.py
git commit -m "feat: per-difficulty and per-class scoring report with CLI"
```

---

## Task 7: Upload, and settle the matching rule

**Files:** none — this is an experiment whose result is written back to `.context/`.

- [ ] **Step 1: Upload `out/submission-empty.json`** to the portal Benchmark page.
      Record the marks awarded per difficulty. Submissions are unlimited and best-run-stands,
      so this cannot cost anything.

- [ ] **Step 2: Record the floor.** The difference between the awarded marks and our local
      `proxy_score` of 0.00 **is** the floor component. Write it into
      `.context/07-platform-and-scoring.md`.

- [ ] **Step 3: Probe the matching rule.** Build a submission containing exactly one D2 event:
      ground truth for T028 has a `traffic_accident` at **30–35 s**. Submit a prediction of
      **41–46 s** — 6 s late, so tIoU = 0 but both endpoints are within 15 s.
      - If it scores → the 15 s tolerance is an independent acceptance path.
        Set `boundary_tolerance_sec=15.0` as the project default.
      - If it does not → tIoU ≥ 0.5 is the only rule. Leave the default strict.

- [ ] **Step 4: Write the finding** into `.context/06-decisions.md` with the date, the
      observation, and what would reverse it.

- [ ] **Step 5: Commit**

```bash
git add .context
git commit -m "docs: record portal floor and the resolved D2/D3 matching rule"
```

---

## Definition of done

- [ ] `pytest -v` is green, including the integration tests, with the dataset present.
- [ ] `python scripts/make_empty_submission.py` writes a file that `validate_submission` accepts.
- [ ] `python scripts/score_submission.py out/submission-empty.json` reports 0 false alarms
      and 20/18/8 misses.
- [ ] Scoring ground truth against itself yields `proxy_score == 100.0`.
- [ ] The empty submission has been uploaded and its marks recorded in `.context/`.
- [ ] The D2/D3 matching rule is resolved and recorded, or explicitly noted as still unknown.
