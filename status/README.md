# Status — read this first

**Updated 14:10 IST · build closes 18:00 · ~3h50m left**

## Where we are in one line

Infrastructure is done and proven and the demo page works on real output, but the restarted Gemini
run is writing failures to disk as predictions and cannot be scored until it is fixed. **No scored
result yet, and nothing uploaded beyond the all-empty file.**

## Live right now

| # | Experiment | Model | Where | State |
|---|---|---|---|---|
| B | Zero-shot, 34 public videos | Gemini 3.5 Flash | Google API | running 14/34, **output unreliable — see below** |
| A | Zero-shot, 34 public videos | Qwen3-VL-4B-Instruct | Modal A100-40GB | running, output file still empty |
| D | Self-hosted Cosmos-Reason2-8B | Cosmos-Reason2-8B | Modal A100 | queued |
| G | LoRA fine-tune | Qwen3-VL-4B | Modal A100-80GB | **blocked** on training-data regeneration |
| — | Demo review page | — | `demo/index.html` | **working**, rebuilds from any run in one command |

## What is actually proven

- Modal works end to end: A100-SXM4-40GB, model loads, 34-video pipeline runs, output written.
- The submission JSON passes the validator; the scoring/merge/sweep chain runs on CPU with no GPU.
- Frame sampling is safe on the awkward clips (1.875 fps, 30 total frames → 24 unique frames back).
- The demo page renders truth and prediction on one axis and scores them with the team's own
  matcher. Verified against a real run and against malformed input.

## What is NOT proven

- **No model quality number exists yet.** Both baselines are mid-flight.
- Nothing has been submitted to the portal beyond the all-empty file.
- The fine-tune has not started.

## The restarted Gemini run cannot be read as a result

Its second attempt is recording 10 of its first 14 videos as `events: []` when they actually threw.
`empty_window_raw` is empty for all ten, which only the exception path in `run.py` produces — a
genuine "no events" answer records the model's raw text. Two compounding causes: `maxOutputTokens`
is 384 with no thinking budget set, so a Gemini 3.x reasoning model spends the allowance on
thinking and returns a fragment or nothing; and a failed video is then written with the same bytes
as a normal one. Reported to the owner at 14:12, fix is in their lane. **No number from this run
goes anywhere until it is re-run.**

## Honest early signal — 6 videos, all D1, from the FIRST run

From `out/gemini-partial6.events.jsonl`, the six videos the first attempt completed before it
crashed, scored with `ahc_vad.scoring`:

| | |
|---|---|
| found | 1 (T005 `traffic_accident`) |
| missed | 1 (T006 `traffic_accident`, predicted nothing) |
| false alarm | 1 (T003 `vehicle_blocking_traffic` @0.85 on a video with no anomaly) |
| correct normals | 3 (T001, T002, T004) |

Precision 50%, recall 50% **on six videos of thirty-four, all from the easiest difficulty.**
That is an anecdote with a ratio attached, not a result. False alarms are the dominant failure mode
on this leaderboard — one entrant lost 26 of 35 marks at D2 to 94 of them.

## Blocked on you

| # | What | Why it matters |
|---|---|---|
| 1 | **Make the repo public or host-accessible** | It is pushed but private. The submission spec requires the hosts can read it. |
| 2 | Upload the all-empty submission | Floor ~13.5 marks, zero risk, validates the schema. Confirm whether this is done. |
| 3 | Fill in the portal profile | The leaderboard shows your real name. |
| 4 | **Who renders the two slides, and to what format?** | Draft is in `slides.md`. Highest-weight judged item and it has no owner. |

Cleared since the last update: git access, Modal spend cap ($30), `HF_TOKEN`, read-only portal
access via Chrome.

## Not worth your time

**NVIDIA NIM is dead for today** — diagnosed, not guessed. See [`experiments.md`](experiments.md).
Don't chase a new key; the fix takes days and we don't need it.
