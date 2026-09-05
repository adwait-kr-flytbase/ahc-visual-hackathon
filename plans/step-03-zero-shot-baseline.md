# Step 3 — Zero-Shot VLM Baseline

> ## ✅ STATUS: `BUILT` — implemented in `src/vad/`, do NOT re-implement
>
> This brief was written before the inference session built the code. It exists now and is
> CPU-tested. **Adopt it and add tests, or bin it — but do not build both.**
> Several "open questions" below are already answered by working code; they are kept for the
> record. See [`../HANDOFF.md`](../HANDOFF.md).
>
> Modules: `vad/prompts.py` (ASK-HINT grouped prompt, tolerant parser), `vad/engine.py` (HF or any OpenAI-compatible server), `vad/run.py`

**Goal:** Get a real, honest number on the 34 public videos with **no training at all**, in about an
hour, so every later result has something to be compared against — and so there is a parachute if
fine-tuning does not converge.

**Why it exists:** ASK-HINT (WACV 2026) reports **89.83% AUC on UCF-Crime training-free**, above
several fine-tuned methods. See [`../.context/02-prior-art.md`](../.context/02-prior-art.md). With
the realistic human ceiling on the leaderboard around 50/100, a training-free baseline is genuinely
competitive, not a toy.

**Data used**
- `dataset/test/videos/*.mp4` — the 34 public videos (inference input only)
- `dataset/test/ground_truth.csv` — to score, never to tune

**Rough shape:** an off-the-shelf small VLM, prompted with ASK-HINT-style *grouped, fine-grained*
questions rather than one flat "is anything wrong here" — a traffic group, a hazard group, a
behaviour group — then map answers onto the 11 class strings and emit via Step 1's writer.

**Open questions to close first**
- Which backbone, and served how? Qwen2.5-VL-7B is ASK-HINT's own; Qwen3-VL-4B is the organisers'
  default; Qwen3.5-4B scores far higher on MMMU-Pro but **its video-input support is unverified**
  ([`../.context/08-sota-landscape.md`](../.context/08-sota-landscape.md)) — settle this early, it
  also decides Step 4's backbone.
- Do we run whole short clips at D1 and windows at D2/D3, or one uniform strategy?
- Exact prompt text and the answer→class mapping. This is the whole method; it deserves drafting
  and iterating, not improvising.
- Where do D2/D3 timestamps come from with no training — window index, or does the model emit them?

**Depends on:** Step 1. Independent of Step 2 — **these two can run in parallel**, one is CPU, one GPU.
