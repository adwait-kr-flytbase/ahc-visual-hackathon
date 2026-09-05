# Status — read this first

**Updated 13:50 IST · build closes 18:00 · ~4h10m left**

## Where we are in one line

Infrastructure is done and proven; two zero-shot baselines are running; training is blocked only on
the other agent's data regeneration. **No scored result yet.**

## Live right now

| # | Experiment | Model | Where | State |
|---|---|---|---|---|
| C | Zero-shot, 34 public videos | Qwen3-VL-4B-Instruct | Modal A100-40GB | running |
| B | Zero-shot, 34 public videos | Gemini 3.5 Flash | Google API | running, 3/34 |
| — | Full 15GB dataset → Modal Volume | — | Modal CPU | running, detached |
| D | Self-hosted Cosmos-Reason2-8B | Cosmos-Reason2-8B | Modal A100 | queued |
| G | LoRA fine-tune | Qwen3-VL-4B | Modal A100-80GB | **blocked** — see below |

## What is actually proven

- Modal works end to end: A100-SXM4-40GB, model loads, 34-video pipeline runs, output written.
- The submission JSON passes the other agent's validator.
- The scoring/merge/sweep chain works on CPU with no GPU and no re-inference.
- Frame sampling is safe on the awkward clips (1.875 fps, 30 total frames → 24 unique frames back).

## What is NOT proven

- **No model quality number exists yet.** Both baselines are mid-flight.
- Nothing has been submitted to the portal beyond the all-empty file.
- The fine-tune has not started.

## Honest early signal

First 3 Gemini predictions: T001 correct (normal), T002 correct (normal),
**T003 FALSE POSITIVE — predicted `vehicle_blocking_traffic` at 0.85 confidence on a `normal`
video.** False alarms are the dominant failure mode on this leaderboard (one entrant lost 26 of 35
marks to 94 FAs). Treat any single good example as an anecdote; only the scored 34 counts.

## Blocked on you

| # | What | Why it matters |
|---|---|---|
| 1 | **Permission to `git commit` + push** | 15+ files uncommitted. **The repo is a graded deliverable.** Biggest single risk right now. |
| 2 | Modal spend cap | I'm queuing more GPU jobs on your account. |
| 3 | Chrome permission for the portal domain | So I submit and read the leaderboard myself instead of handing you files. |
| 4 | `HF_TOKEN` in `.env` | Nice-to-have; Cosmos-Reason2-8B is 16GB and an unauthenticated pull can rate-limit. |

## Not worth your time

**NVIDIA NIM is dead for today** — diagnosed, not guessed. See [`experiments.md`](experiments.md).
Don't chase a new key; the fix takes days and we don't need it.
