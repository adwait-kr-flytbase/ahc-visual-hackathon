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
from ahc_vad.submission import validate_submission

ROOT = Path(__file__).resolve().parents[1]


def load_predictions(path: Path) -> tuple[dict, dict[str, list[Event]]]:
    payload = json.loads(path.read_text())
    predictions = {
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
    return payload, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--ground-truth", type=Path,
                        default=ROOT / "dataset" / "test" / "ground_truth.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.json")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance", type=float, default=None,
                        help="seconds; enables the alternative acceptance path")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    payload, predictions = load_predictions(args.submission)

    if not args.skip_validation:
        problems = validate_submission(payload, manifest)
        if problems:
            print(f"!! {len(problems)} schema problem(s) -- the portal may reject this:")
            for problem in problems[:20]:
                print(f"   - {problem}")
            if len(problems) > 20:
                print(f"   ... and {len(problems) - 20} more")
            print()

    report = score(
        load_ground_truth(args.ground_truth),
        predictions,
        manifest,
        policy_d23=MatchPolicy(
            iou_threshold=args.iou, boundary_tolerance_sec=args.boundary_tolerance
        ),
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
