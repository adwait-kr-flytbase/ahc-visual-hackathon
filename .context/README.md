# .context — AHC Visual Intelligence Hackathon

Knowledge base for the FlytBase AHC Visual Intelligence Hackathon
(**Real-Time Video Anomaly Detection**, 05 Sep 2026).

## Files

| File | Contents |
|---|---|
| [`00-problem.md`](00-problem.md) | Problem statement distilled. Constraints, the two core tensions, temporal-scale and false-positive subtleties. **Start here.** |
| [`01-dataset.md`](01-dataset.md) | The data: layout, 12-class label set, `ground_truth.csv` schema, level tiers, and the **contradictions** it creates with the problem statement. **Read second — it redefines the task.** |
| [`02-prior-art.md`](02-prior-art.md) | The 4 organiser-supplied papers + SOTA deck, reading priority, what to extract. Not yet read. |
| [`03-finetuning-tooling.md`](03-finetuning-tooling.md) | Unsloth / ms-swift / TRL+PEFT, organisers' gotchas, implied model size band and default recipe. |
| [`04-compute-and-access.md`](04-compute-and-access.md) | Free GPU runtimes (Kaggle, Colab, Lightning, Modal), hosted APIs, pre-event checklist. |
| [`05-research-agenda.md`](05-research-agenda.md) | Open questions, ordered by decision impact. Answered ones marked. |
| [`09-dataset-profile.md`](09-dataset-profile.md) | **Measured** profile of the downloaded pack — class balance, video metadata, the train/test mismatch, provenance leaks. |
| [`06-decisions.md`](06-decisions.md) | Append-only decision log — what was chosen, why, and what would reverse it. |
| [`07-platform-and-scoring.md`](07-platform-and-scoring.md) | **Submission JSON spec, the 3 difficulties, the 100-mark scoring, the live leaderboard, and the fact that the practice-pack answer key ships with the dataset.** Read before writing any output code. |
| [`08-sota-landscape.md`](08-sota-landscape.md) | **2026 SOTA survey beyond the 4 papers**: streaming/token-budget VLMs, Qwen3.5 small series, Cosmos 3, the AI City Challenge TAR track, what enterprises ship, and the still-open ego-motion gap. |
| [`09-poc-plan.md`](09-poc-plan.md) | **The build plan**: full system in 5 layers, cloud placement, ordered build queue for the remaining hours, and the cut order. |
| `sources/` | Verbatim copies of the four original organiser docs. |
| `artifacts/` | Files pulled from the platform: `manifest.json`, `submission-template.json`. |

## Implementation plans

Specs live in [`../plans/`](../plans/README.md). `step-01-scoring-harness.md` is `READY`;
steps 2–7 are `DRAFT` briefs and must be discussed before any code is written.

## Status

- [x] Problem statement read and distilled
- [x] Primer + prerequisites read and distilled
- [x] Dataset doc read and distilled
- [x] Dataset downloaded (15 GB, `dataset/`) — train class folders + `test/` with 34 videos and ground truth
- [x] Level 1/2/3 semantics resolved — see `07-platform-and-scoring.md`
- [x] Submission format confirmed — JSON spec read off the portal, template downloaded
- [x] Prior-art papers read (all 4) and framework docs checked
- [x] Participant portal explored (all 6 pages) and leaderboard read
- [x] 2026 SOTA landscape surveyed (`08-sota-landscape.md`)
- [ ] SOTA deck reviewed
- [ ] Local re-implementation of the scoring function
- [ ] Baseline all-normal submission uploaded to validate the pipeline
- [x] Approach chosen (see `../plans/`)
- [ ] Step 1 implemented

## The framing, revised after the dataset doc

> A **scored competition** with a private evaluation set. Given CCTV / dashcam / drone video,
> detect and temporally localise events from a **fixed 12-class taxonomy**, using a model small
> enough to run in real time on limited GPU, with precision high enough to stay usable.

Not the open-ended "detect anything anomalous" build the problem statement implied.

## Working hypothesis (provisional — see `06-decisions.md`)

**Cerberus-style cascade**: cheap always-on gate → VLM verifier, with ASK-HINT-style grouped
fine-grained prompts as the day-one training-free baseline. Fine-tuning via **ms-swift + Qwen3-VL-4B**
(Unsloth is images-only; the task is video).

**The known gap:** every paper in the reading list evaluates on *fixed-camera CCTV*. Our data is
one-third drone, one-third dashcam. Cerberus's cheap gate is frame-differencing, which assumes a
static background. **Ego-motion is the main adaptation risk and the likeliest differentiator.**

## Open questions blocking any approach decision

1. ~~What are levels 1, 2 and 3?~~ **Resolved.** D1 = 24 short clips, class only, no timing, 25 marks.
   D2 = 6 × 240 s, class + timing, 15 s boundary tolerance, 35 marks. D3 = 4 long (308–629 s)
   multi-event videos, 40 marks. See `07-platform-and-scoring.md`.
2. ~~What is the submission format, and how is it scored?~~ **Resolved.** One JSON file, per-video
   `events[]` (empty = normal) + self-reported `runtime_metadata`; D2/D3 events count only at
   class-match **and temporal IoU ≥ 0.5**. Best run stands, unlimited submissions.
3. ~~Is the closed 12-class set the whole story?~~ **Resolved.** The submittable taxonomy is
   **11 anomaly classes**; `normal` is expressed as an empty `events` array, never as a class string.
4. **How exactly are the 25/35/40 marks decomposed?** Two entrants score 1.8 + 11.7 + 0.0 with zero
   found and zero FA, so there is a floor/normal-credit component we have not reverse-engineered.
   Worth pinning down before optimising, because it sets the value of a conservative low-recall policy.
5. **Is the private eval set drawn like the public one?** The public L2/L3 videos look synthetically
   composed on regular time grids. If the private set shares that construction it changes what
   temporal post-processing is worth building.
6. **What earns the reasoning bonus?** The benchmark page says "not graded"; the leaderboard shows
   `+3.5` and `+1.0`. `explanation` is 20–500 chars and can never reduce the score, so always emit it.

## Immediate next actions

1. Fill in the **Profile / player card** — it is blank and the leaderboard uses the real name.
2. Replicate the scoring function locally against `dataset/test/ground_truth.csv`.
3. Upload a well-formed **all-normal** submission to validate the schema end to end (worth ~13.5/100
   and cannot lower a later best run).
4. Profile the train split: class balance, description coverage, durations.
