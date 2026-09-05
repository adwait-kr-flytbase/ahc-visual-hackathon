"""Local event matching and scoring.

This is a *matcher*, not a replica of the portal's marks formula -- the floor component of
25/35/40 is not yet reverse-engineered. For iteration only the ranking has to be faithful.
See .context/07-platform-and-scoring.md.
"""

from dataclasses import dataclass

from ahc_vad.events import Event, temporal_iou
from ahc_vad.groundtruth import VideoInfo

DIFFICULTY_MARKS = {1: 25.0, 2: 35.0, 3: 40.0}


@dataclass(frozen=True)
class MatchPolicy:
    """How a predicted event is allowed to satisfy a ground-truth event.

    iou_threshold           minimum temporal IoU (the documented D2/D3 rule)
    boundary_tolerance_sec  if set, an ALTERNATIVE acceptance path: both endpoints within
                            this many seconds. The portal documents a 15 s tolerance as well
                            as IoU >= 0.5 and it is unclear how they combine -- keep this
                            configurable until a probe upload settles it.
    require_temporal        False for D1, where only the class is scored.
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
    by_class = {name: DifficultyScore(0, *counts) for name, counts in sorted(per_class.items())}
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


# ---------------------------------------------------------------------------
# Per-video scoring -- the portal's ACTUAL metric.
#
# Measured 2026-09-05 from a real upload: an all-empty submission scored
# D2 = 11.7/35 = 33.3%, and exactly 2 of the 6 D2 videos are normal. 2/6 = 33.3%.
# So marks are (mean per-video correctness) x (max marks for that difficulty), NOT the
# global event-level F1 that `score()` above computes.
#
# The consequence is large and counter-intuitive:
#     D1  24 videos / 25 marks  =  1.04 marks per video
#     D2   6 videos / 35 marks  =  5.83 marks per video
#     D3   4 videos / 40 marks  = 10.00 marks per video
# One D3 video is worth ten D1 videos. Optimise accordingly.
#
# How partial credit works WITHIN a multi-event video is not yet known -- D1 came back
# 12.1/25 = 48.4%, which is not a multiple of 1/24, so some partial credit exists.
# `per_video_credit` is therefore pluggable: probe the portal, then pick the one that matches.
# ---------------------------------------------------------------------------


def credit_f1(result: MatchResult) -> float:
    """Per-video credit = F1 over that video's events. Partial credit for partial finds."""
    tp, fa, miss = result.true_positives, result.false_alarms, result.misses
    if tp == 0:
        return 1.0 if (fa == 0 and miss == 0) else 0.0
    precision = tp / (tp + fa)
    recall = tp / (tp + miss)
    return 2 * precision * recall / (precision + recall)


def credit_exact(result: MatchResult) -> float:
    """Per-video credit = 1 only if every event matched and nothing was invented."""
    return 1.0 if (result.false_alarms == 0 and result.misses == 0) else 0.0


def credit_recall(result: MatchResult) -> float:
    """Per-video credit = recall, ignoring false alarms entirely.

    Worth probing: the leaderboard shows entrants ABOVE us at D1 carrying MORE false alarms
    (14 found / 8 FA beats 10 found / 3 FA), which is what this would predict.
    """
    tp, miss = result.true_positives, result.misses
    if tp + miss == 0:
        return 1.0 if result.false_alarms == 0 else 0.0
    return tp / (tp + miss)


def score_per_video(
    gt: dict[str, list[Event]],
    pred: dict[str, list[Event]],
    manifest: dict[str, VideoInfo],
    *,
    policy_d1: MatchPolicy | None = None,
    policy_d23: MatchPolicy | None = None,
    credit=credit_f1,
) -> dict:
    """Estimate portal marks: mean per-video credit x max marks, per difficulty."""
    policy_d1 = policy_d1 or MatchPolicy(require_temporal=False)
    policy_d23 = policy_d23 or MatchPolicy()

    per_level: dict[int, list[float]] = {level: [] for level in DIFFICULTY_MARKS}
    per_video: dict[str, float] = {}
    for video_id, info in manifest.items():
        result = match_events(
            gt.get(video_id, []),
            pred.get(video_id, []),
            policy_d1 if info.level == 1 else policy_d23,
        )
        value = credit(result)
        per_level[info.level].append(value)
        per_video[video_id] = value

    marks, total = {}, 0.0
    for level, values in per_level.items():
        mean = sum(values) / len(values) if values else 0.0
        awarded = mean * DIFFICULTY_MARKS[level]
        marks[level] = {
            "videos": len(values),
            "mean_credit": round(mean, 4),
            "marks": round(awarded, 2),
            "max": DIFFICULTY_MARKS[level],
            "marks_per_video": round(DIFFICULTY_MARKS[level] / len(values), 2) if values else 0.0,
        }
        total += awarded
    return {"total": round(total, 2), "by_difficulty": marks, "per_video": per_video}
