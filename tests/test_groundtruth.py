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
    assert load_ground_truth(path) == {"T029": []}


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
    assert not load_ground_truth(path)["T005"][0].is_localised


def test_commas_inside_the_description_do_not_break_parsing(tmp_path):
    path = _write(tmp_path, '''
        video_id,level,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary
        T001,1,true,fire,1,2,"A fire, with smoke, spreads"
    ''')
    assert load_ground_truth(path)["T001"][0].explanation == "A fire, with smoke, spreads"


def test_train_csv_without_a_level_column_still_loads(tmp_path):
    path = _write(tmp_path, """
        video_id,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary
        TR01350,true,traffic_accident,0.000,5.000,A traffic collision occurs.
    """)
    assert load_ground_truth(path)["TR01350"][0].span == (0.0, 5.0)


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
    assert by_level == {1: 20, 2: 18, 3: 8}
