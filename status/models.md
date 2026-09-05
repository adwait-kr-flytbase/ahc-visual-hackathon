# Model selection — what we chose, what we compared it against, and why

**Updated 15:05 IST.** Every model decision on this build, with the evidence behind it.

---

## The constraint that filters everything

The problem statement is explicit: *"Larger hosted models can be used during development, for
comparison, or to generate training data, but they cannot be part of what makes the detector work at
runtime."*

So any candidate must be **(a) self-hostable, (b) small enough to run in real time on limited GPU,
and (c) able to take video or multi-image input.** That eliminates most of the obvious answers
before any measurement happens.

| Rejected before testing | Why |
|---|---|
| GPT-5.6, Claude, Gemini Pro | Hosted. Cannot be in the runtime path. |
| Qwen2.5-VL-72B, Llama-3.2-90B | Too large for real-time on limited GPU. |
| Llama-3.2-11B/90B-Vision | **Accepts one image per request.** We send 8–32 frames. Structurally unusable for video. |
| Full fine-tuning of anything | Does not fit the VRAM or the clock. |

---

## The head-to-head, measured

**All on the identical 24-video D1 set, same ASK-HINT grouped prompt, 16 frames/window, top-1
output policy, scored with `ahc_vad.scoring`. Modal A100.**

| Model | Params | P | R | F1 | Silent videos |
|---|---|---|---|---|---|
| **Qwen3-VL-4B-Instruct** | 4B | **0.77** | **0.50** | **0.61** | 8/24 |
| Qwen3-VL-8B-Instruct | 8B | 0.62 | 0.40 | 0.48 | 8/24 |

*(4B vs 8B differ on only 3 of 24 videos — statistically a tie. See the retraction below.)*
| Cosmos-Reason2-8B | 8B | 0.57 | 0.20 | 0.30 | 13/24 |
| Qwen3-VL-2B-Instruct | 2B | 0.30 | 0.15 | 0.20 | 12/24 |
| Qwen2.5-VL-7B-Instruct | 7B | 0.00 | 0.00 | 0.00 | **20/24** |

### ⚠️ Retracted: "bigger is worse" is NOT supported

I initially reported that 4B beats 8B as a finding. **It does not survive scrutiny and I have
withdrawn it.**

The two models differ on **3 videos out of 24**: 4B wins T007 (accident, 8B silent) and T014
(smoke, 8B says fire); both lose T011. A sign test on 2 informative trials gives **p = 0.25**.
The F1 gap of 0.61 vs 0.48 comes entirely from 10 TP vs 8 TP — inside the ±1 TP ≈ 5pp noise band
this document states everywhere else.

It is **not** a measurement artifact — both models emit clean `{"events": []}`, zero truncation,
zero parse failures, and *the same number of empty windows (11 each)*. They are equally willing to
answer; they simply disagree on 3 videos.

**The defensible claim: Qwen3-VL-4B and 8B are statistically indistinguishable on this test.**
We choose the 4B because it is **half the size at equal measured accuracy**, which is the right call
under a real-time constraint and needs no scaling claim to justify it.

### Two findings that DO survive

**1. Benchmark leadership does not transfer.**

Cosmos-Reason2-8B **tops NVIDIA's own Traffic Anomaly Reasoning leaderboard** (AI City Challenge 2026
Track 3) and is purpose-built for physical-AI reasoning. On this task it gets **half the recall of a
4B general model**. Reasoning-QA benchmark performance does not predict short-clip anomaly
classification. This is the single most surprising result of the build.

**2. Qwen2.5-VL-7B is silent on every video.**
Not a parsing failure — it returns a bare `[]` (rather than the requested `{"events": [...]}`)
24 times out of 24. Notably it is **ASK-HINT's own published backbone**, on which that paper reports
89.83% AUC on UCF-Crime. It does not reproduce on aerial/dashcam footage with our prompt. Its
different output shape also suggests weaker instruction-following than the Qwen3 line.

### The decision

**Qwen3-VL-4B-Instruct.** Chosen on measurement, not default. It was also the organisers' own
example model, but we only kept it after four alternatives lost to it.

---

## What the rest of the field chose

Read from the live leaderboard (31 entrants). Ignoring the 100.0 — almost certainly the shipped
answer key — the genuine field is **entirely two approaches**:

| Approach | Entrants | Best |
|---|---|---|
| SigLIP2 / CLIP + a temporal head | Yash 65.9, Ruturaj 60.2, Rejoy 59.5, Shreyas 57.0, Sarthak 47.1, Daksh 30.9 | **65.9** |
| Small Qwen-VL + LoRA | Aryan 51.1, Manikandan 47.8, Aditya 36.4, Yuvraj 33.6, Yogender 33.1, Revanth 29.4 | **51.1** |

**The frozen-embedding + trained-head approach is beating every fine-tuned VLM on the board**, and it
does so with 4 of the top 5 places. That is not noise — it is several independent teams converging.

Mechanism, from `.context/02-prior-art.md`: Alert-CLIP (CVPR 2026) measured that CLIP's normal-vs-
abnormal *text* embeddings are entangled, so raw CLIP similarity is a poor discriminator — and their
tuning widens the margin from <0.16 to >0.38. A **tuned** similarity head is therefore much stronger
than raw CLIP prompting, which is likely where the ceiling on that approach comes from.

**Why we are not simply copying it:** a CLIP+head is ~100× cheaper per frame than a 4B VLM, which
makes it the natural **cheap always-on stage of a cascade**, not a competitor to the VLM. That is the
Cerberus structure and it is the hackathon's actual question. It is still untried on our side — see
below.

---

## Still untried, and honestly why

| Candidate | Status | Reason |
|---|---|---|
| **SigLIP2 + temporal head** | **Not tried — the biggest gap in our coverage** | The leading approach on the board. Needs Kaggle T4 or a spare A100 slot. I originally marked it ⛔ on the theory that a probe cannot use context; the leaderboard is evidence and the theory was wrong. |
| **Qwen3.5-4B** | Not tried | Released 2026-03, natively multimodal, **MMMU-Pro 65.4 vs Qwen3-VL-4B's 52.0**. Video-input support unverified. Would be the obvious next backbone if it takes video. |
| InternVL / MiniCPM-V / SmolVLM | Not tried | No reason to expect them to beat Qwen3-VL-4B given 8B and 7B already lost to it. Low expected value. |
| Ensemble of Qwen3-VL-4B + 8B | Not tried | Both runtime-legal. A precision play, but our problem is recall. |

---

## Reference ceilings — dev-time only, never submittable

| Model | Verdict |
|---|---|
| **Gemini 3.5 Flash** | Works, video-native. **No trustworthy number** — two runs invalidated by our own bugs (a 384-token budget consumed by Gemini 3.x thinking tokens; free-tier 429 storms). |
| NVIDIA NIM catalogue | **Dead for this account.** 4 of 5 models 404 — personal orgs ship without the "Public API Endpoints" entitlement, which takes days to resolve. The one working model takes a single image. |

---

## The fine-tune, and why it is the only lever left

Five inference-time interventions on the frozen model have now failed: frame count (8/16/24/32), two
prompt variants, and four alternative models. **Nothing moved the number.**

What that establishes is narrow but real: *zero-shot* Qwen3-VL-4B is blind to the context and
duration classes (`stalled` 0/1, `blocking` 0/1, `road_spill` 0/2, `fighting` 0/2) while confident on
appearance classes (`accident` 3/3, `smoke` 2/2, `flood` 2/2). It does **not** establish that a VLM
cannot learn them — that is precisely the untested sixth intervention.

**Running now:** LoRA rank 16, frozen ViT + frozen aligner, 5,633 rows (short clips + synthetic long-
video windows + negatives), `FPS_MAX_FRAMES=24`, `max_steps 400`, `save_steps 200` on an A100-80GB.
