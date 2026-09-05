# Plans

Implementation specs for the AHC Visual Intelligence Hackathon build.
**Background and evidence live in [`../.context/`](../.context/README.md) — read that first.**

## How to use this folder

Each step is one spec file. A spec is only safe to implement when its **Status** is `READY`.

| Status | Meaning |
|---|---|
| `DONE` | Implemented and committed. |
| `BUILT` | **Code exists already — do not re-implement from spec.** Needs tests and review only. |
| `READY` | Fully specified, safe to execute. |
| `DRAFT` | A brief only. Open questions must be closed first. |

> **Process note, 2026-09-05 13:15 — the original bar has been dropped.**
> This folder originally required every spec to be executable "by an engineer with no prior
> context", red-green-commit TDD throughout. That is a multi-day standard and the build closes at
> 18:00. From here: **build directly with tests alongside, and spec only what is genuinely unclear.**
> Steps 3/5/6/7 are already implemented in `src/vad/` — they are `BUILT`, not `DRAFT`.

## Steps

| # | Spec | Status | Depends on | Marks at stake |
|---|---|---|---|---|
| 1 | [Scoring harness & submission emitter](step-01-scoring-harness.md) | ✅ **DONE** (Task 7 pending portal) | — | Unlocks all measurement; ~13.5 floor |
| 2 | [Long-video synthesizer](step-02-long-video-synthesizer.md) | **IN PROGRESS** | 1 | Unlocks 75 |
| 3 | [Zero-shot VLM baseline](step-03-zero-shot-baseline.md) | **BUILT** (`src/vad/`) | 1 | Establishes the number to beat |
| 4 | [SFT on short clips (D1)](step-04-sft-short-clips.md) | `DRAFT` — **blocked on GPU** | 1, 2 | 25 |
| 5 | [Long-video inference & merge (D2/D3)](step-05-long-video-inference.md) | **BUILT** (`src/vad/`) | 1, 2, 4 | 75 |
| 6 | [Precision & threshold tuning](step-06-precision-tuning.md) | **BUILT** (`src/vad/`) | 1, 2, 5 | 5–15 recovered |
| 7 | [Instrumentation, demo & deck](step-07-instrumentation-and-deck.md) | **BUILT** (`src/vad/`) | 5 | Latency bonus + host judging |

## Conventions for every spec in this folder

**Language & layout.** Python 3.11+. Source in `src/ahc_vad/`, tests in `tests/`, one-shot
executables in `scripts/`. `pyproject.toml` puts `src/` on the path for pytest — there is no
install step.

**Testing.** pytest. Every task follows red → green → commit. Tests that touch `dataset/` are
marked `@pytest.mark.integration` and skipped when the pack is absent, so the suite runs anywhere.

**Determinism.** Any function that ranks, matches, or samples must break ties by a stable key
(index order), never by set/dict iteration order. Scores must be reproducible run-to-run.

**No fitting to the public test set.** `dataset/test/` ships with its own answer key. It is a
*measuring instrument*, never a training or tuning target. See
[`../.context/06-decisions.md`](../.context/06-decisions.md).

**Provenance in comments.** When a constant comes from an observed fact, cite it:
`# 11 classes, not 12 — see .context/07-platform-and-scoring.md`.

**Commits.** Small and frequent, conventional-commit style (`feat:`, `test:`, `fix:`, `docs:`).
A **public code repository is itself a graded deliverable** (`.context/07-platform-and-scoring.md`
§Step 2), so history quality counts.

## Ground rules an executing agent must not violate

1. **Never emit `normal` as a class.** Absence of anomaly is `"events": []`. 11 classes, not 12.
2. **Levels come from `data/manifest.json`**, never from a ground-truth CSV — `train/` has no
   `level` column at all.
3. **All time is in seconds as float**, never frame indices. Source frame rates span 1.875–30 fps.
4. **If a spec is `DRAFT`, stop and ask.** Do not infer the missing half.
5. **Ownership is by directory. Cross-boundary changes are requested, not made.**
   `src/ahc_vad/`, `plans/`, `scripts/`, `tests/` — this session. `src/vad/`, `bootstrap_gpu.sh` —
   the inference session. See [`../HANDOFF.md`](../HANDOFF.md).
6. **Never hand-roll the training prompt.** Build SFT rows with
   `from vad.prompts import build_sft_sample`. If the training and inference templates drift by one
   token the fine-tuned model scores worse than zero-shot. Highest-risk integration point in the build.
