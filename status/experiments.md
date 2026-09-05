# Experiment log

Every model tried, what happened, and the verdict. Newest first within each section.

---

## Models in play

| Model | Size | Role | Runtime-legal? |
|---|---|---|---|
| **Qwen3-VL-4B-Instruct** | 4B | The submission model — zero-shot now, LoRA fine-tune next | ✅ yes |
| **Cosmos-Reason2-8B** | 8B | Self-hosted challenger; tops the Traffic Anomaly Reasoning leaderboard | ✅ yes |
| Gemini 3.5 Flash | hosted | **Reference ceiling only** | ❌ no — hosted models are banned from the runtime path |
| Qwen3.5-4B | 4B | Candidate backbone, much stronger on MMMU-Pro (65.4 vs 52.0) | ✅ if video input confirmed |

## A · Zero-shot, Qwen3-VL-4B — RUNNING

Modal A100-40GB. 16 frames/window, 20s window, 10s hop. Frames sampled by timestamp.
Smoke test on 3 videos passed: model loaded, 0 events on T001–T003, which is correct (all normal).
**Result pending.** This is the number the fine-tune has to beat.

## B · Zero-shot, Gemini 3.5 Flash — RUNNING

Native `generateContent` API, 10 frames/window, JSON response mode, exponential backoff on 429/503.
**REFERENCE CEILING ONLY — cannot be submitted.** A hosted model cannot be in the runtime path.
Its value is telling us how much of the gap is *prompting* vs *model capacity*.

Partial, 3/34: T001 ✓ normal · T002 ✓ normal · **T003 ✗ false positive, `vehicle_blocking_traffic`
@0.85 on a normal video.**

## C · NVIDIA NIM — DEAD, diagnosed

Text-ping probe on the user's key:

| Model | HTTP |
|---|---|
| `meta/llama-3.2-11b-vision-instruct` | **200** |
| `nvidia/vila` | 404 |
| `nvidia/cosmos-reason2-8b` | 404 |
| `microsoft/phi-3-vision-128k-instruct` | 404 |
| `nvidia/nemotron-nano-3-30b-a3b` | 404 |

**Root cause:** personal NVIDIA orgs ship *without* the "Public API Endpoints" permission. Models
appear in `GET /v1/models` and work in the web playground, but `POST /v1/chat/completions` returns
404 "Function not found for account". Widely reported; the only fix is an NVIDIA developer-forum
request, which takes **days**.

The one enabled model, Llama-3.2-11B-Vision, accepts **one image per request** and we send 8–16
frames — it 400s. Dead end either way.

**Why it doesn't matter:** `nvidia/Cosmos-Reason2-8B` is **ungated on HuggingFace** (verified, HTTP
200) and we have an A100. NIM's only value was hosting models we couldn't run ourselves. We can.
→ Experiment D.

## D · Self-hosted Cosmos-Reason2-8B — QUEUED

8B in bf16 ≈ 16GB, fits A100-40GB. Purpose-built for physical-AI reasoning and tops the traffic
anomaly leaderboard. Runtime-legal, unlike Gemini. Runs as soon as the GPU frees up.

## E · Merge-policy sweep — READY, costs nothing

72 configs over `min_conf` × `gap_tol` × `min_dur`, re-scored on CPU with **no re-inference**,
because raw per-window predictions are saved separately from merged events. With a 15s boundary
tolerance and IoU≥0.5, merge policy is worth roughly as much as model quality — and it's free.

## F · LoRA fine-tune — BLOCKED on data

Command is written and will be run **verbatim**; the image already has `ms-swift`.
Notable departures from the organisers' defaults, all deliberate:
`FPS_MAX_FRAMES 16→24` (16 over a 20s window is 0.8fps and can't see a 5s accident),
`VIDEO_MAX_TOKEN_NUM 128→256`, `VIDEO_MAX_PIXELS→448×448`, `max_length 4096→8192`, `lora_rank 8→16`.

**Blocked because:** the organisers relabelled `wrong_way_driving` mid-event (108 of 164 rows →
`normal`). The other agent's clip pool keyed on *directory name* rather than *class name*, so it
would have spliced ordinary traffic into videos labelled as wrong-way events — poisoning the class
that has exactly one example in the public test set. Caught and fixed; data regenerating, ~8 min.

---

## Silent failures caught today

None of these crash. All four would have shown up as "the model just isn't very good."

1. **ffmpeg truncates `-ss/-t`** past a clip's real length without error — a 468s video rendered as
   241s and every subsequent timestamp was wrong.
2. **49 train rows have `end_time_sec <= start_time_sec`** — garbage spans, not exceptions.
3. **Clip pool keyed on folder, not label** — would have poisoned `wrong_way_driving`.
4. **Empty model output vs. correct "no events"** were indistinguishable in my own code — now every
   window that yields zero events records the model's raw text.
