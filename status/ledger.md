# Adaptation ledger — every way to make a model better, and where we stand

**Updated 14:15 IST.** The complete space of things you can do to a VLM, ordered from cheapest to
most expensive. Status on each. This is the running record of what we tried, what worked, what
didn't, and what we deliberately skipped.

Legend: ✅ done, measured · 🔄 running · ⬜ planned · ⛔ rejected, with reason · ❌ tried, failed

---

## Tier 0 · Input space — no weights touched, no training, instant

| # | Technique | Status | Result |
|---|---|---|---|
| 0.1 | **Timestamp-based frame sampling** (not frame indices) | ✅ | Required. Clip fps ranges 1.875–30; index-based sampling would silently mis-time every event. |
| 0.2 | **Frame count per window** (8 / 16 / 24 / 32) | ⬜ | 16 in use. Measured: even 30-frame clips return 24 *unique* frames, so 24 is safe. **The single dial most likely to matter** — 16 over a 20s window is 0.8 fps and cannot resolve a 5s accident. |
| 0.3 | **Resolution** (448 vs 640 max edge) | ⬜ | 640 in use for inference, 448 for training. Untested as a variable. |
| 0.4 | **Window size / hop** (20s/10s vs 10s/5s) | ⬜ | 20/10 in use. A 10s hop quantises boundaries to 10s, which cannot satisfy IoU≥0.5 on a 5s event. Suspected D2 ceiling. |
| 0.5 | **Motion-mask prompting** — overlay red circles/squares on moving regions to steer attention (Cerberus) | ⬜ | Free, no training, reported >50% frame reduction at 95% recall. Assumes a static camera; needs ego-motion compensation for drone/dashcam. |
| 0.6 | **Frame tiling** — 9 frames into one 3×3 grid image | ⛔ | Was the workaround for single-image models (Llama 3.2 Vision). Moot once NVIDIA access failed. |

## Tier 1 · Prompting — no weights touched

| # | Technique | Status | Result |
|---|---|---|---|
| 1.1 | **ASK-HINT grouped fine-grained prompting** (traffic / hazard / behaviour groups, discriminative cue per class) | ✅ | In use. **D1: P=0.77 R=0.50, 13/24 correct, 3 FA.** Precision better than anyone on the leaderboard. |
| 1.2 | **Describe-then-classify** (caption first, then decide) | ⬜ | Both Cerberus and TAU-R1 route through captions. Our `explanation` field is half of this already. |
| 1.3 | **Few-shot in-context exemplars** | ⬜ | Costly in tokens — each exemplar is another set of frames. Likely poor value at 16 frames/window. |
| 1.4 | **Self-consistency** — sample N times, vote | ⬜ | N× the cost for a precision gain. Cheap alternative: vote across *prompts* rather than samples. |
| 1.5 | **Per-class prompt specialisation** for the 5 classes the model is silent on | ⬜ | **Highest-value untried prompting idea** — see the failure analysis below. |

## Tier 2 · Decoding and output policy — no weights, no training, huge effect

| # | Technique | Status | Result |
|---|---|---|---|
| 2.1 | **Top-k output policy** | ✅ | **MEASURED, and it matters more than expected.** D1 top-1: P=0.77. top-2: P=0.53. all events: P=0.45 — *with identical recall*. The model names the right class first, then adds 1–3 spurious ones. Every extra class is a pure false alarm. |
| 2.2 | **Per-class confidence thresholds** | ⬜ | Swept by `vad.sweep` at zero GPU cost. Rare classes (1 test example each) may be worth suppressing entirely. |
| 2.3 | **Merge policy** — `gap_tol`, `min_dur`, `min_conf` | ⬜ | 72-config sweep ready, runs on CPU with no re-inference. With a 15s tolerance this is worth roughly as much as model quality. |
| 2.4 | **Constrained / guided JSON decoding** | ⬜ | vLLM `guided_json` makes malformed output structurally impossible. Currently handled by a tolerant parser instead. |
| 2.5 | **Temporal persistence** — fire only if N of M windows agree | ⬜ | Classic false-alarm suppressor. Untried. |

## Tier 3 · Frozen backbone + trained head — cheap, and what the leaderboard is doing

| # | Technique | Status | Result |
|---|---|---|---|
| 3.1 | **CLIP/SigLIP2 embeddings + linear probe** | ⬜ **reopened** | **The leading approach on the board — 4 of the top 5, best 65.9, above every fine-tuned VLM.** I originally marked this ⛔ on the theory that a probe cannot use context. The leaderboard is evidence and the theory was wrong. Reopened as the biggest gap in our coverage. It is ~100× cheaper per frame than a 4B VLM, so it is the natural **cheap stage of a cascade**, not a competitor. |
| 3.2 | **Temporal head (GRU / transformer) over frame embeddings** | ⛔ | Same reasoning. `siglip-gru-stage-a` scores 47.1 on the board. |
| 3.3 | **Alert-CLIP representation tuning** — widen the normal/abnormal margin from <0.16 to >0.38 | ⛔ | Correct fix for a genuine CLIP defect, but needs region annotations we don't have and a training budget we don't have. |

## Tier 4 · Parameter-efficient fine-tuning — our main bet

| # | Technique | Status | Result |
|---|---|---|---|
| 4.1 | **LoRA on language layers, frozen ViT + frozen aligner** | 🔄 | Command ready, runs verbatim on a Modal A100-80GB. Blocked only on training data. rank 16, alpha 32, all-linear, lr 1e-4, 1 epoch. |
| 4.2 | **Token-budget dials** — `FPS_MAX_FRAMES` 16→24, `VIDEO_MAX_TOKEN_NUM` 128→256, `VIDEO_MAX_PIXELS`→448² | 🔄 | Deliberate departures from the organisers' defaults. Justified: 16 frames over 20s is 0.8 fps. |
| 4.3 | **QLoRA** (4-bit frozen base) | ⬜ | Fallback if A100-80GB OOMs. Costs some quality. |
| 4.4 | **Unfreezing the vision encoder** | ⛔ | Every organiser example freezes it. Aerial/dashcam viewpoints arguably justify tuning it, but not on this clock. |
| 4.5 | **DoRA / IA³ / prefix tuning** | ⛔ | No evidence they beat LoRA here, and each is a new failure surface. |
| 4.6 | **Full fine-tune** | ⛔ | Doesn't fit the VRAM or the clock. |

## Tier 5 · Preference and RL post-training

| # | Technique | Status | Result |
|---|---|---|---|
| 5.1 | **GRPO with a tIoU-shaped reward** (class correctness + tIoU + format validity) | ⬜ | **The genuinely state-of-the-art move for this task.** STVG-R1, VTG-Reasoner, Video-R1/T-GRPO, TAU-R1 and COPRA all show SFT alone underperforms on temporal grounding. 75 of 100 marks are IoU-gated. Too expensive for today — it is the "what we'd do next" slide. |
| 5.2 | **DPO on preferred outputs** | ⬜ | Cheaper than GRPO, needs preference pairs we don't have. |

## Tier 6 · Distillation

| # | Technique | Status | Result |
|---|---|---|---|
| 6.1 | **Response distillation** — large VLM generates labels, small VLM trains on them | ⬜ | Explicitly sanctioned ("larger models may generate training data"). Blocked by Gemini rate limits; the data is already labelled anyway, so this is an upgrade not a rescue. |
| 6.2 | **Rationale distillation** — train on the big model's reasoning traces | ⬜ | Would feed the `explanation` bonus directly. |
| 6.3 | **Logit distillation** | ⛔ | Requires a shared tokenizer with the teacher. Not available. |

## Tier 7 · Cascades and ensembles

| # | Technique | Status | Result |
|---|---|---|---|
| 7.1 | **Cheap gate → VLM verifier** (Cerberus) | ⬜ | The reference architecture; 57.68 fps at 1% anomaly rate. Its gate is frame-differencing, which assumes a static camera. |
| 7.2 | **Codec motion vectors + global-motion fit as an ego-motion-robust gate** | ⬜ | **Our original idea.** Motion vectors come free out of the H.264 decoder; a homography fit over the MV field *is* the ego-motion, and the residual is independently-moving objects. Also a D2/D3 temporal proposal generator. Nothing in the field is doing this. |
| 7.3 | **Multi-model vote** (Qwen + Cosmos) | ⬜ | Both runtime-legal. Precision play. |
| 7.4 | **Class-routed hybrid** — send each class to whichever mechanism measures better | ⬜ | **The headline idea.** VLM for appearance classes, track/motion state for duration classes. Routing table derived from a controlled experiment, not intuition. **Methodological rule: derive the routing on `synth-dev`, report on `dataset/test`** — routing tuned on the data we report is not a result. |
| 7.5 | **Track-state layer** (YOLO11n + ByteTrack, 4 fps, ego-compensated speed) | 🔄 | Agent 2-26. `stalled` >8s stationary, `loitering` >12s, `congestion` density↑ + median speed <45% of baseline. `wrong_way` and `blocking` deliberately excluded from v1 — one public test example each, worst marks-per-hour on the board. |

---

## What the D1 failure analysis actually says

Measured on 24 videos, top-1 policy:

**Works:** `traffic_accident` 3/3 · `smoke` 2/2 · `waterlogging_or_flood` 2/2 · `fire` 1/2 · `traffic_congestion` 1/2
**Silent — model returns nothing at all:** `road_spill_or_debris` 0/2 · `fighting_or_violence` 0/2 · `stalled_or_broken_down_vehicle` 0/1 · `vehicle_blocking_traffic` 0/1
**Wrong class:** `wrong_way_driving` → called it an accident

The model is confident on **appearance** classes (fire, smoke, flood, crash wreckage) and mute on
**context and duration** classes (a car stopped too long, a vehicle obstructing, debris on a
carriageway, people fighting). That is exactly the split the problem statement predicted — and it is
precisely what fine-tuning on our data should fix, because those classes are well represented in
train (223 stalled, 151 spill, 148 blocking, 124 fighting).

### ⚠️ Precision about what the six experiments actually falsified

**All five failed interventions were inference-time changes to a *frozen* model.** So what is
established is:

- ✗ falsified — *"it can't see the event"* (more frames is worse)
- ✗ falsified — *"it sees it but won't commit"* (both prompt variants are worse)
- **not tested** — *"the model can learn this"*

The defensible claim is **"zero-shot Qwen3-VL-4B is blind to duration classes"**, NOT "VLMs cannot
express duration". The sixth intervention — fine-tuning on synthetic long videos with
window-relative timestamps for sustained classes — is precisely the untested one, and it is the one
we are about to run. If it moves those four classes, the stronger claim would be something we
published and then contradicted.

*(Correction raised by agent 2-26, accepted. I had over-generalised from five inference-time
experiments to a claim about model capability.)*

**Two cheap things to try before the fine-tune finishes:** per-class prompts for the five silent
classes (1.5), and a lower confidence floor for them specifically (2.2). Neither costs GPU.

## Models evaluated → **see [`models.md`](models.md) for the full selection rationale**

| Model | Params | P | R | F1 | Verdict |
|---|---|---|---|---|---|
| **Qwen3-VL-4B-Instruct** | 4B | **0.77** | **0.50** | **0.61** | **Chosen.** Beat four alternatives. |
| Qwen3-VL-8B-Instruct | 8B | 0.62 | 0.40 | 0.48 | ❌ Bigger is worse, same family. |
| Cosmos-Reason2-8B | 8B | 0.57 | 0.20 | 0.30 | ❌ Tops NVIDIA's Traffic Anomaly Reasoning leaderboard; half the recall of a 4B here. |
| Qwen3-VL-2B-Instruct | 2B | 0.30 | 0.15 | 0.20 | ❌ Too small. |
| Qwen2.5-VL-7B-Instruct | 7B | 0.00 | 0.00 | 0.00 | ❌ Silent on 20/24. ASK-HINT's own backbone; does not reproduce here. |

Reference ceilings (dev-time only, never submittable): Gemini 3.5 Flash — works, video-native, but
no trustworthy number (two runs invalidated by our own bugs). NVIDIA NIM — 4 of 5 models 404 for
this account; the one that works takes a single image.

**NVIDIA NIM — actionable:** the primer says *"Verify your phone number when prompted — API access
stays locked until you do."* Only 1 of 5 models we probed responds. **Phone verification at
build.nvidia.com is the likely fix** and takes two minutes. Worth trying once; if models still 404
it is the known org-level "Public API Endpoints" entitlement, which takes days and is not worth
chasing.

**Also unclaimed:** the primer describes an *AI Grants India × FlytBase* form, given out on
hackathon day, which returns an OpenAI `gpt-5.6-luna` key (~4 RPM). We have never requested it.
