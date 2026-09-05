# Platform, submission format and scoring

Source: **measured** — read directly from the participant portal on 2026-09-05, plus the
downloaded `manifest.json` / `submission-template.json` (copies in `.context/artifacts/`).

Portal: `https://fbhackathonplatform-production.up.railway.app/participant/p_a_K-Df`
(Google-SSO, `adwait.kalsekar@flytbase.com`, in-person participant.)

## Pages

| Page | Path | Contents |
|---|---|---|
| Home | `/participant/<id>` | HUD: next step, current score, run count |
| The Quest | `…/quest-map` | Difficulty briefs + event schedule. *"Your host hasn't written the brief yet."* |
| Benchmark | `…/benchmark` | **Submission format spec, manifest/template downloads, upload form, Step-2 final submission** |
| My Submissions | `…/submissions` | Run history; best run counts |
| Public Arena | `/arena` | Live leaderboard + per-difficulty P/R/found/FA breakdown |
| Profile | `…/register` | Player card (name, role, phone; email locked to SSO) |

**Profile is not filled in yet** — the leaderboard uses the real name, so this should be completed.

## Scoring — 100 marks, one file, three difficulties

| Difficulty | Videos | Durations | Task | Marks |
|---|---|---|---|---|
| **D1 · Clear event** | 24 (T001–T024) | 5.7 – 26.1 s | Is anything happening + which class. **Timing not scored.** | **25** |
| **D2 · When it happens** | 6 (T025–T030) | all exactly 240 s | Class + start/end. **Boundary tolerance 15 s** | **35** |
| **D3 · Long context** | 4 (T031–T034) | 307.7 / 360 / 376.5 / 628.8 s | Same as D2 over long footage, multi-event | **40** |

- Total pack: 34 videos, **3391 s ≈ 56.5 min** — this *is* the public test set from the Drive mirrors.
- **An event counts at D2/D3 only if the class is right AND temporal overlap (IoU) ≥ 0.5.**
- A difficulty never answered scores **zero**. Videos never answered are scored **as normal**.
- Submissions are incremental: a file only updates the video ids it mentions. **Best run stands** —
  a worse attempt never costs you. Unlimited submissions.
- **Reasoning bonus** sits in its own column, on top of D3. Portal says *"not graded"* on the
  benchmark page, but the leaderboard shows non-zero values (`+3.5`, `+1.0`) — it is being computed.
- **Latency bonus** is computed from the `end_to_end_internal_time_ms` you self-report per video.

### Step 2 — final submission (judged by hosts, separate from the benchmark score)
1. **Code repository** (public or host-accessible) — required
2. **Final presentation — exactly two slides** — required, *"carries high weight in the final judging"*.
   Cover: what you built · approach and why · what you learned. Prefer graphs/tables/timelines/example
   frames over paragraphs. Call out anything that made it faster, cheaper, more reliable, or lower-FA.
3. Anything else — optional

## Submission JSON

Top level: `schema_version`, `submission_id`, `model_name`, `run_metadata` (optional:
`total_wall_time_ms`, `max_parallel_videos`, `hardware`), `predictions[]`.

Per prediction: `video_id`, `events[]`, `runtime_metadata`.

| Field | Required | Rule |
|---|---|---|
| `video_id` | yes | Exact manifest id, at most once |
| `events` | yes | **Empty array = you predict normal.** Never emit `"normal"` as a class |
| `class_name` | per event | One of **11** anomaly classes (see below) |
| `start_time_sec` | per event | `null` at D1; required and ≥ 0 at D2–D3 |
| `end_time_sec` | per event | `null` at D1; must exceed start and stay inside duration |
| `explanation` | optional | 20–500 chars. Bonus only, **never reduces score** |
| `runtime_metadata` | yes | per video: `frames_processed`, `chunks_processed`, `end_to_end_internal_time_ms`, `model_runtimes[]` |

`model_runtimes[]` entries: `model_name`, `call_count`, `total_time_ms`, `average_time_ms`,
`p50_time_ms`, `p95_time_ms`, `max_time_ms`.

### Taxonomy — **11 classes, not 12**
`traffic_accident` · `traffic_congestion` · `stalled_or_broken_down_vehicle` ·
`vehicle_blocking_traffic` · `fire` · `smoke` · `waterlogging_or_flood` · `wrong_way_driving` ·
`road_spill_or_debris` · `fighting_or_violence` · `loitering_or_suspicious_presence`

`normal` is **not** a submittable class — it is expressed as `events: []`.
(The dataset doc's "12 classes" counts `normal` as a folder; the submission taxonomy is 11.)

### `manifest.json`
`{schema_version, videos:[{video_id, level, domain, duration_sec}]}` — **`domain` is empty for every
video**, so CCTV/dashcam/drone is not given and must be inferred if we want it.

## Leaderboard (read 2026-09-05 ~11:57 IST, 7 entrants)

Columns exposed per difficulty: **marks · P · R · found · FA**. This leaks the answer-key sizes:

- D1: **20 of 24 videos are anomalous** (4 normal)
- D2: **18 events** total
- D3: **8 events** total

| # | Participant | model_name | D1 /25 | D2 /35 | D3 /40 | Marks | Bonus |
|---|---|---|---|---|---|---|---|
| 1 | Yash Waghmare | `probe` (24 runs) | 25.0 (P100 R100 20/20, 0 FA) | 29.9 (P100 R22 4/18) | 37.2 (P100 R50 4/8) | 92.1 | – |
| 2 | Aryan Varale | `qwen3vl4b-lora-finetuned-tuned` (15) | 10.6 | 25.1 | 12.0 | 47.6 | +3.5 |
| 3 | Neeraj Gupta | `cascade-appearance-vlm` (3) | 17.5 | 23.7 | 8.0 | 49.2 | – |
| 4 | Manikandan Vinayakan | `Qwen2.5-VL-3B-Instruct + lora` (12) | 13.2 | 8.9 (94 FA) | 11.2 (67 FA) | 33.2 | +1.0 |
| 5 | Aniruddha More | `qwen3-vl-4b-lora` (6) | 2.6 | 15.2 | 2.0 | 19.8 | – |
| 6–7 | M B Thejesshwar / Jay Kelani | `my-model` (3) | 1.8 | 11.7 | 0.0 | 13.5 | – |

**What the board tells us:**
- `probe` is a host sanity-check run (100% precision everywhere), not a real competitor. **Realistic
  human ceiling right now is ~50/100.** The field is wide open.
- **False alarms are brutal.** Manikandan has the 2nd-best D1 recall but 94 FA at D2 → 8.9/35.
  Neeraj's 8 FA at D1 costs him ~7 marks despite 70% recall.
- Two entrants scoring 1.8/25 + 11.7/35 with **zero found and zero FA** ⇒ **there are free marks for
  submitting nothing but empty `events` arrays** (the normal videos score, plus a floor per difficulty).
  A well-formed all-normal file is worth ~13.5/100 and is the correct first upload to validate the pipeline.
- Everyone at the top is on a **small Qwen-VL (3B–4B) with LoRA**, which matches our tooling decision.

---

## ⚠️ The practice pack's answer key ships with the dataset

`dataset/test/ground_truth.csv` (from the Drive mirrors) has **52 rows over exactly T001–T034**, and
its event counts match the leaderboard's denominators exactly: **L1 20 anomalous / 24**, **L2 18
events**, **L3 8 events**. The portal's "held-out ground truth" for the *practice pack* is therefore
already in our hands.

**Consequence:** the public leaderboard is not a measure of anything. Do **not** treat a high
practice score as progress, and do not fit to it. Its real uses are:
1. **Validating the submission pipeline** (schema, ids, timing units) end to end.
2. As a **labelled dev set** for honest offline evaluation — with the scoring function replicated
   locally so we can iterate without uploading.

The ranking that matters is the **private evaluation set** plus the host-judged code + two-slide deck.

### Public-test composition (measured from `dataset/test/ground_truth.csv`)

| Level | Videos | Normal videos | Events |
|---|---|---|---|
| L1 | 24 | 4 (T001–T004) | 20, one per video |
| L2 | 6 | 2 (T029, T030) | 18 across T025–T028 |
| L3 | 4 | 0 | 8 across T031–T034 |

Class distribution over all 52 rows: `traffic_accident` 16 · `traffic_congestion` 7 ·
`loitering_or_suspicious_presence` 7 · `normal` 6 · `road_spill_or_debris` 3 ·
`fighting_or_violence` 3 · `vehicle_blocking_traffic` 2 · `fire` 2 · `smoke` 2 ·
`waterlogging_or_flood` 2 · `stalled_or_broken_down_vehicle` 1 · `wrong_way_driving` 1.
**Heavily accident-skewed; `stalled_or_broken_down_vehicle` and `wrong_way_driving` have one example each.**

Structural notes on the long videos — these look **synthetically composed** from short clips:
- T025: six `traffic_accident` events on an exact 40-s grid (20–40, 60–80, … 220–240).
- T028: four `traffic_accident` events on an exact 60-s grid, each exactly 5 s long.
- T026: four events of **four different classes** in one 240-s video.
- T027: four `traffic_congestion` segments, one of them 60 s long.
- T031: a single 125-s `traffic_congestion` occupying the last third of a 360-s video.
- T033: two long `traffic_accident` events (75 s and 45 s) inside 629 s.

37 of 52 rows carry a non-blank `description_summary`.
