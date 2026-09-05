# POC plan — the full system, and what actually fits

Written 2026-09-05 12:10 IST. **Build window closes 18:00 → ~5 h 50 m remain.**
The architecture below is the "go all out" version; §Build order marks what fits today.

---

## L0 · Scoring harness — *no GPU, build first, ~20 min*

**Why local at all, given unlimited free uploads:** the portal can only score the 34 public videos.
It **cannot score the ~100 synthetic long videos** from L1.3 — and those are the only D2/D3
supervision that exists. That alone settles it. Secondary: the board returns 5 aggregate numbers per
difficulty, with no per-video or per-class attribution; and per-class threshold sweeps need hundreds
of evaluations, which is not an upload workflow.

**What is actually needed is the *matcher*, not a faithful marks replica** — greedy class-match +
tIoU ≥ 0.5 → precision / recall / FA per difficulty. The absolute 25/35/40 number only matters on the
leaderboard; for iteration only the *ranking* has to be faithful. (An exact replica is impossible
anyway until the floor component is reverse-engineered.) On the 34 public videos, local scoring is
equivalent to uploading — the answer key is already in hand; the portal's only unique value there is
confirming the JSON parses.

| Piece | What |
|---|---|
| `score.py` | Greedy event matcher: D1 class-only, D2/D3 class-match **and tIoU ≥ 0.5**, 15 s boundary tolerance → P / R / FA per difficulty, broken out per class and per video. |
| `submit.py` | Emits schema-valid JSON (11 classes, `events: []` = normal, `runtime_metadata` per video). |
| Dev sets | (a) the 34 public test videos — **the answer key ships in `dataset/test/ground_truth.csv`**; (b) synthetic long videos from L1.3. |

Unknown: the exact decomposition of the 25/35/40 and the non-zero floor two entrants scored with
zero found and zero FA. **Reverse-engineer it with probe uploads** — unlimited, best-run-stands,
so a probe can never cost marks.

**First upload: a well-formed all-normal file.** Worth ~13.5/100, validates the schema end-to-end.

## L1 · Data factory — *CPU, embarrassingly parallel → Modal fan-out*

1. **`index.parquet`** — ffprobe all 3,173 clips: duration, fps, resolution, bitrate.
   fps ranges **1.875 → 30**, so every sampler works in **timestamps, never frame indices**.
2. **Domain tagging** — `manifest.json` ships `domain` empty for every test video. Tag all clips
   drone / dashcam / CCTV with zero-shot SigLIP2 or CLIP over 3 frames. Enables stratified eval and
   viewpoint-conditioned prompts. Validate by eyeballing 50.
3. **⭐ Long-video synthesizer — the highest-leverage component in the whole build.**
   Train has no long or multi-event videos; D2+D3 is 75 marks of exactly that (see `01-dataset.md`).
   Concatenate train clips + `normal` filler into 240 s and 300–630 s videos with exact ground truth,
   ffmpeg-concat re-encoded to a fixed 25 fps / 720p. Mirror the observed public-test compositions
   **and** randomise beyond them:
   - fixed 40 s / 60 s grids (T025, T028 pattern)
   - four different classes in one 240 s video (T026 pattern)
   - one long event occupying a third of a 360 s video (T031 pattern)
   - randomised onsets, variable gaps, 0-event (all-normal) long videos as negatives
   Output: a few thousand D2/D3 training samples + ~100 held-out synthetic long dev videos.
4. **Description / CoT enrichment** — `description_summary` is already 100% populated, so this is
   an upgrade, not a rescue. Gemini 2.5 Flash (takes video directly, free tier) or NVIDIA NIM
   (~40 RPM) → structured `{visual_evidence, key_objects, onset_cue, why_anomalous}`.
   Feeds the `explanation` bonus field and the describe-then-decide intermediate.
5. **Hard negatives** — the 973 `normal` clips, plus mined near-misses: parked-in-a-bay vs.
   stopped-on-a-shoulder, crowd vs. fight, dusk haze vs. smoke.

## L2 · Cheap evidence layer — *GPU-light, always-on, ego-motion-robust*

6. **⭐ Codec motion-vector ego-motion decomposition.** `ffmpeg -flags2 +export_mvs` gives
   per-macroblock motion vectors **free out of the H.264 decoder**. RANSAC-fit an affine/homography
   per frame pair → that fit *is* the ego-motion; the **residual field is independently-moving
   objects**. Replaces Cerberus's frame-differencing gate, which assumes a static camera and breaks
   on our drone + dashcam thirds. Per-second outputs: residual energy, spatial spread, blob count.
   **Doubles as the D2/D3 temporal proposal generator.**
7. **Detection + tracking** — RT-DETR or YOLO-World (open-vocab) + ByteTrack at 5 fps. Per track:
   class, dwell time, ego-compensated speed, heading. Directly yields:
   `stalled` (speed≈0 beyond N s) · `wrong_way` (heading vs. fitted dominant flow) ·
   `congestion` (density × falling mean speed) · `blocking` (stationary vehicle + queue behind) ·
   `loitering` (person dwell) · `fighting` (person-pair proximity + pose motion energy).
   Nothing in the reading list keeps per-object state; the AI City winners all reached for a tracker.
8. **Appearance cues, non-VLM** — `fire`: red-orange saturation + high-frequency flicker.
   `smoke`: low-saturation grey blob, expanding boundary, local contrast collapse.
   `waterlogging_or_flood`: large low-texture specular region, horizontal reflectance.
   Three classes largely solvable without a VLM.
9. **Evidence timeline** — one row per second per video, ~20 numeric features + track summaries.
   Shared substrate for proposals, thresholds, and the demo overlay.

## L3 · VLM core — *GPU*

10. **Zero-shot baseline, day-zero, no training** — Qwen2.5-VL-7B (ASK-HINT's own backbone) or
    Qwen3-VL-4B with **ASK-HINT grouped fine-grained prompts**: traffic group
    (`accident`/`congestion`/`stalled`/`blocking`/`wrong_way`), hazard group
    (`fire`/`smoke`/`flood`/`spill`), behaviour group (`fighting`/`loitering`).
    89.83% AUC on UCF-Crime with 6 prompts — a genuinely competitive fallback.
11. **SFT — ms-swift LoRA, frozen ViT + frozen aligner.** Train the model to **emit the submission
    JSON directly** (constrained decoding at inference), on three mixed sources:
    short clips → `{class, start, end, explanation}` · synthetic long videos → multi-event list ·
    `normal` clips → `{"events": []}`.
    Backbone A/B: **Qwen3-VL-4B-Instruct** (organiser default, native video in ms-swift) vs
    **Qwen3.5-4B** (MMMU-Pro 65.4 vs 52.0, Apache 2.0 — *but video input unverified*) vs Qwen2.5-VL-7B.
    **Sweep, do not accept, the token dials:** `FPS_MAX_FRAMES=16` is ~0.5 fps on a 30 s clip and
    cannot see a 5 s accident cleanly; `VIDEO_MAX_TOKEN_NUM=128` is very tight. These are the
    binding constraint on temporal localisation.
12. **Long-video inference** — sliding windows (~20 s window, 10 s hop) restricted to L2-proposed
    regions; VLM emits per-window class + local span; a merge state machine unions overlapping
    same-class windows into events. The 15 s D2 tolerance makes this forgiving.
13. **Stretch — GRPO with a tIoU-shaped reward** (class correctness + tIoU + format validity).
    Known to beat SFT on temporal grounding (STVG-R1, VTG-Reasoner, T-GRPO, COPRA).
    **Not today.**

## L4 · Fusion and precision control

14. **Proposal → verification cascade.** L2 proposes `(t_start, t_end, class prior)` at high recall;
    the VLM verifies at high precision. Cerberus's precision/recall split, with an ego-robust gate.
15. **Per-class thresholds tuned against the local scorer, not one global threshold.**
    False alarms are brutal on this board (94 FA → 8.9/35 despite good recall).
    `stalled_or_broken_down_vehicle` and `wrong_way_driving` have **one test example each** — the
    scorer-optimal policy may be to suppress rare classes unless highly confident. That is a
    tuning decision, not a modelling one.
16. **Event lifecycle** — merge same-class events separated by < X s; de-duplicate persistent
    conditions so they fire once, not per frame.

## L5 · Serving, instrumentation, demo

17. vLLM (or SGLang) serving the LoRA-merged model; batched offline path for the leaderboard.
18. **Instrumented streaming path** producing honest `model_runtimes[]` (call_count, p50, p95, max)
    → this is what earns the **self-reported latency bonus**, and it is the demo.
19. Demo artifact: video player with the L2 evidence timeline and live alerts overlaid.

---

## Cloud setup

| Job | Where | Why |
|---|---|---|
| L1 data factory, L2 evidence extraction | **Modal**, CPU fan-out over 3,173 clips | per-second billing, $30/mo credit already on file, trivially parallel |
| SFT | **Modal A100-80GB / H100**, or **Lightning L40S** | Qwen3-VL-4B LoRA ≈ 2×21 GB at default dials — does **not** fit one T4 |
| Backup / overnight trainer | **Kaggle T4×2**, 30 GPU-h/week free | free, but needs reduced `FPS_MAX_FRAMES` / `VIDEO_MAX_PIXELS` |
| Label enrichment | Gemini 2.5 Flash (video-native) + NVIDIA NIM (~40 RPM) | dev-time only — must not be in the runtime path |

Persist to a Modal Volume: `index.parquet`, `evidence/*.parquet`, `synth/*.mp4`, checkpoints.

---

## Build order for the ~5 h 50 m that remain

| # | Task | Est. | Marks at stake |
|---|---|---|---|
| 1 | L0 matcher + all-normal submission | 20 m | validates everything; ~13.5 |
| 2 | L1.3 long-video synthesizer (Modal fan-out) | 45 m | unlocks 75 |
| 3 | L3.10 zero-shot ASK-HINT baseline on the 34 test videos | 45 m | real starting number |
| 4 | L3.11 SFT on short clips → D1 | 90 m | 25 |
| 5 | L3.11+12 SFT on synthetic long videos + merge | 90 m | 75 |
| 6 | L4.15 per-class threshold tuning on the local scorer | 30 m | 5–15 recovered from FA |
| 7 | L5.18 instrumentation + two-slide deck | 30 m | bonus + "high weight in final judging" |

**Run 2 and 3 in parallel** — one is CPU on Modal, the other is GPU.

### Cut order, when the clock bites
GRPO (13) → tracking (7) → domain tagging (2) → Gemini enrichment (4) → appearance heuristics (8)
→ motion-vector gate (6, keep only if D2/D3 boundaries are missing).

**Irreducible core:** local scorer · long-video synthesizer · ms-swift SFT emitting submission JSON ·
sliding-window merge. Everything else is upside.

### Open risks
- **Qwen3.5-4B video input is unverified.** If it is image-only, fall back to Qwen3-VL-4B and lose
  the MMMU-Pro gap. Check this in the first 10 minutes — it decides the SFT backbone.
- The private eval set may **not** share the public set's synthetic grid composition. If it is real
  continuous footage, the synthesizer's distribution is wrong and boundary accuracy will drop.
  Mitigate by randomising onsets and gaps rather than copying the observed grids exactly.

---

# Review — 2026-09-05, against the measured profile (`09-dataset-profile.md`)

Plan is sound: L0 first, the synthesizer as the centrepiece, and the MV gate for ego-motion are the
three right calls. Gaps below, ranked by how much score they put at risk.

## 🔴 1. The synthesizer will teach the model to detect **scene cuts**, not events

**Measured:** for 7 of 11 classes the labelled event *is the entire clip* — `loitering` and
`waterlogging` 100%, `fighting` 94%, `fire` 91%, `congestion` 90%, `smoke` 82%, `accident` 77%.

So a naive concat gives **event boundary ≡ clip boundary ≡ the encoding seam**. A model trained on
that learns "a hard cut with changed lighting/compression = event onset". It will look excellent on
synthetic dev videos and collapse on D3, which is continuous real footage with no cuts at the event
boundaries (T031's congestion *develops*; T033's collisions occur mid-shot).

**This is the single biggest failure mode in the plan, and it is invisible without a real dev set.**

Fixes, all cheap:
- **Crop events to sub-spans.** Place a clip so the event occupies a *random interior* portion —
  pad both sides with normal filler *and* with lead-in/lead-out from the same clip where duration allows.
- **Put distractor cuts inside normal filler**, at the same rate as at event boundaries, so a cut
  carries zero information about the label.
- **Hold out real D3 as the only trusted boundary metric.** Synthetic dev tIoU will read optimistically.

## 🔴 2. The synthesizer bakes in the *training* format, not the *test* format

Plan says "re-encoded to a fixed **25 fps / 720p**". That is the train distribution (85.5% is
1280×720, 75% is 15 fps). **Test D1 is 640×640, 720×404, 896×448, 256×192; test fps is 24 / 30 /
29.97 / 1.875.** 640×640 is a square crop that appears nowhere in train.

Fixing the synthesizer to 25 fps/720p makes the gap *worse*. **Randomise resolution, aspect and fps
across the synthetic set**, sampling from the test-side distribution, and letterbox rather than
stretch. Note also that re-encoding to 25 fps destroys the 1.875 fps texture of the *only* source
corpus that supplies `loitering` (300/300 clips) and the test `fighting`/`loitering` videos.

## 🟠 3. Duration is a leaky class prior — concatenating whole clips imports it

**Measured:** `waterlogging` = 95 clips, 5 distinct durations, all ≈5.7 s. `loitering` = 300 clips,
**3 distinct durations**, all ≈30 s. `fire`/`smoke` p50 = 5.8 s.

Concatenate whole clips and every synthetic `waterlogging` event is 5.7 s and every `loitering`
event is 30 s. The model learns duration → class. Real D3 event durations are
**2.6 / 9.4 / 13.7 / 19.5 / 37.6 / 45.0 / 75.0 / 125.0 s** — nothing like that.
**Vary event span independently of source-clip length.**

## 🟠 4. No clip-level train/val split *before* synthesis

The plan holds out ~100 synthetic long dev videos but never splits the 3,173 source clips. If dev
videos are built from clips also used in training, the dev set is leaked and every number from it is
inflated. **Split clips first, synthesise train and dev from disjoint pools.**

## 🟠 5. §15's threshold tuning contradicts §L0's "don't fit the public set"

Tuning per-class thresholds on 34 videos — where `stalled` and `wrong_way` have **one example each**
— is fitting to the public set, which the plan elsewhere correctly warns against. The public pack is
accident-skewed (16 of 52 rows); the private set almost certainly is not. **"Suppress rare classes
unless highly confident" is exactly the decision that is optimal on n=1 and potentially catastrophic
privately.** Tune thresholds on held-out *synthetic* data, sanity-check on public, never the reverse.

## 🟡 6. Implementation gotchas that will cost time

- **ffmpeg concat demuxer requires identical codec parameters.** Train has **23 distinct resolutions
  and 10 distinct frame rates**. Naive `-f concat` will fail or silently corrupt. Normalise each clip
  (scale + pad + fps filter) to the target *before* concatenating.
- **No audio streams exist anywhere in the pack** — video-only. Rules out audio fusion entirely.
- **1.875 fps clips are frame-starved**: a 16 s test clip holds **30 frames total**. Any sampler
  requesting a fixed frame count must degrade gracefully rather than error.

## 🟡 7. Resolve the D2/D3 matching rule before tuning boundaries

`07-platform-and-scoring.md` records both a **15 s boundary tolerance** and **tIoU ≥ 0.5**. Those are
different criteria and it is unclear whether they are alternatives, conjunction, or tiers. On a 5 s
event (seven of the eighteen D2 events) 15 s tolerance is trivially satisfied while tIoU ≥ 0.5 is
strict — the two rules differ by a lot. **Settle it with an early probe upload**; it changes whether
to predict tight or generous spans.

## 🟡 8. Build order has no slack
Listed estimates total **375 min** against ~350 min remaining, closed only by running steps 2 and 3
in parallel. Zero buffer for debugging. Given coding agents do the typing this is survivable, but
steps 4 and 5 (90 min each) are the ones that will overrun — protect them by finishing L0 and the
synthesizer early.

## ⚪ Minor
- **Filename collision:** `09-dataset-profile.md` and `09-poc-plan.md` share a number. Rename the
  plan to `10-poc-plan.md`.
- Modal's $30/mo credit is roughly 8–10 h of A100-80GB. Two or three full SFT runs will exhaust it —
  keep Lightning (~$30) and Kaggle T4×2 as live fallbacks, not theoretical ones.

## What the plan gets right and should not be traded away
- L0 before anything else, and the reasoning for local scoring (the portal cannot score synthetic data).
- The synthesizer as the highest-leverage component — correct, and correctly identified as unlocking 75 marks.
- Codec motion vectors for an ego-motion-robust gate — the one genuinely novel idea here, and it
  addresses the gap no paper in `02-prior-art.md` closes.
- Track-state for the four duration-defined classes.
- Constrained decoding to the 11 exact class strings.
