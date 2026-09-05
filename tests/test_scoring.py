from pathlib import Path

import pytest

from ahc_vad.events import Event
from ahc_vad.groundtruth import VideoInfo, load_ground_truth, load_manifest
from ahc_vad.scoring import MatchPolicy, format_report, match_events, score

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"

D1 = MatchPolicy(require_temporal=False)
D23 = MatchPolicy(iou_threshold=0.5)


def test_d1_matches_on_class_alone():
    result = match_events([Event("fire", None, None)], [Event("fire", None, None)], D1)
    assert len(result.matched) == 1
    assert result.unmatched_gt == [] and result.unmatched_pred == []


def test_d1_wrong_class_is_one_miss_and_one_false_alarm():
    result = match_events([Event("fire", None, None)], [Event("smoke", None, None)], D1)
    assert result.matched == []
    assert result.unmatched_gt == [0] and result.unmatched_pred == [0]


def test_exact_span_match():
    gt = [Event("traffic_accident", 20.0, 40.0)]
    assert len(match_events(gt, list(gt), D23).matched) == 1


def test_span_below_the_iou_threshold_does_not_match():
    gt = [Event("traffic_accident", 20.0, 40.0)]
    pred = [Event("traffic_accident", 35.0, 55.0)]
    result = match_events(gt, pred, D23)
    assert result.matched == [] and result.unmatched_pred == [0]


def test_span_just_above_the_iou_threshold_matches():
    gt = [Event("traffic_accident", 20.0, 40.0)]
    pred = [Event("traffic_accident", 25.0, 45.0)]
    assert len(match_events(gt, pred, D23).matched) == 1


def test_right_span_but_wrong_class_never_matches():
    gt = [Event("traffic_accident", 20.0, 40.0)]
    assert match_events(gt, [Event("fire", 20.0, 40.0)], D23).matched == []


def test_one_prediction_cannot_claim_two_ground_truth_events():
    gt = [Event("traffic_accident", 20.0, 40.0), Event("traffic_accident", 60.0, 80.0)]
    result = match_events(gt, [Event("traffic_accident", 20.0, 40.0)], D23)
    assert len(result.matched) == 1 and result.unmatched_gt == [1]


def test_greedy_matching_prefers_the_higher_iou_pair():
    gt = [Event("traffic_accident", 20.0, 40.0)]
    pred = [Event("traffic_accident", 22.0, 44.0), Event("traffic_accident", 20.0, 40.0)]
    result = match_events(gt, pred, D23)
    assert [pi for _, pi, _ in result.matched] == [1]
    assert result.unmatched_pred == [0]


def test_boundary_tolerance_accepts_what_iou_rejects():
    gt = [Event("traffic_accident", 30.0, 35.0)]
    pred = [Event("traffic_accident", 41.0, 46.0)]
    assert match_events(gt, pred, MatchPolicy(iou_threshold=0.5)).matched == []
    lenient = MatchPolicy(iou_threshold=0.5, boundary_tolerance_sec=15.0)
    assert len(match_events(gt, pred, lenient).matched) == 1


def test_empty_prediction_against_normal_video_is_clean():
    result = match_events([], [], D23)
    assert result.matched == [] and result.unmatched_gt == [] and result.unmatched_pred == []


def test_prediction_on_a_normal_video_is_a_false_alarm():
    assert match_events([], [Event("fire", 1.0, 2.0)], D23).unmatched_pred == [0]


def test_matching_is_deterministic_across_repeated_calls():
    gt = [Event("traffic_accident", 20.0, 40.0), Event("traffic_accident", 21.0, 41.0)]
    pred = [Event("traffic_accident", 20.5, 40.5), Event("traffic_accident", 20.6, 40.6)]
    first = match_events(gt, pred, D23)
    for _ in range(20):
        assert match_events(gt, pred, D23) == first


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
    assert score(gt, dict(gt), _manifest()).by_difficulty[1].f1 == 1.0


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
