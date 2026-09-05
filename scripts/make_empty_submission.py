"""Emit an all-empty submission: every video predicted normal.

This is the correct first upload -- it validates the schema end to end and, per the
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
    print(f"Wrote {args.out} -- {len(payload['predictions'])} videos, 0 events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
