# Plans

Implementation specs for the AHC Visual Intelligence Hackathon build.
**Background and evidence live in [`../.context/`](../.context/README.md) — read that first.**

## How to use this folder

Each step is one spec file. A spec is only safe to implement when its **Status** is `READY`.

| Status | Meaning |
|---|---|
| `READY` | Fully specified. An engineer with no prior context can execute it top-to-bottom. |
| `DRAFT` | A brief only. **Do not implement.** Open questions must be closed with the user first. |
| `BLOCKED` | Waiting on a named dependency or decision. |

## Steps

| # | Spec | Status | Depends on | Marks at stake |
|---|---|---|---|---|
| 1 | [Scoring harness & submission emitter](step-01-scoring-harness.md) | **READY** | — | Unlocks all measurement; ~13.5 floor |
| 2 | [Long-video synthesizer](step-02-long-video-synthesizer.md) | `DRAFT` | 1 | Unlocks 75 |
| 3 | [Zero-shot VLM baseline](step-03-zero-shot-baseline.md) | `DRAFT` | 1 | Establishes the number to beat |
| 4 | [SFT on short clips (D1)](step-04-sft-short-clips.md) | `DRAFT` | 1 | 25 |
| 5 | [Long-video inference & merge (D2/D3)](step-05-long-video-inference.md) | `DRAFT` | 1, 2, 4 | 75 |
| 6 | [Precision & threshold tuning](step-06-precision-tuning.md) | `DRAFT` | 1, 2, 5 | 5–15 recovered |
| 7 | [Instrumentation, demo & deck](step-07-instrumentation-and-deck.md) | `DRAFT` | 5 | Latency bonus + host judging |

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
