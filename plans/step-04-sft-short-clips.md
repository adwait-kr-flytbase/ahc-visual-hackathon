# Step 4 — Supervised Fine-Tuning on Short Clips (D1)

> ## ⚠️ STATUS: `DRAFT` — DO NOT IMPLEMENT
>
> This is a **brief**, not a spec. It records intent and the open questions, nothing more.
> Before any code is written, the open questions below must be closed with the user and this
> file rewritten to the standard of [`step-01-scoring-harness.md`](step-01-scoring-harness.md):
> exact files, exact interfaces, real test code, bite-sized TDD steps, no placeholders.
>
> **If you are an agent and you were told to "do step 4", stop and ask instead.**

**Goal:** Teach a small VLM the 11 classes from the short training clips. This is D1 — **25 marks** —
and it is the backbone every later step reuses.

**Data used**
- `dataset/train/<class>/videos/*.mp4` — 2,200 anomaly clips
- `dataset/train/normal/videos/*.mp4` — 973 normal clips, so it learns to say nothing
- `ground_truth.csv` `description_summary` — **100% populated in train**, giving free
  video→text pairs for describe-then-decide

**Rough shape:** LoRA fine-tune with frozen vision encoder and frozen aligner (the organisers'
own recipe), training the model to emit submission-shaped JSON directly, with constrained decoding
onto the 11 exact class strings at inference.

**Known traps** (measured, [`../.context/09-dataset-profile.md`](../.context/09-dataset-profile.md))
- **Train on fixed-length windows, not whole clips.** Clip duration leaks the class —
  `waterlogging` ≈5.7 s always, `loitering` ≈30 s always.
- **Do not accept ms-swift's video defaults.** `FPS_MAX_FRAMES=16` and `VIDEO_MAX_TOKEN_NUM=128` are
  the binding constraint; 16 frames over a 30 s clip is ~0.5 fps and cannot see a 5 s accident.
- **1.875 fps clips are frame-starved** — a 16 s clip holds 30 frames total. Samplers must degrade,
  not error.
- **Unsloth is images-only**; ms-swift has native video input
  ([`../.context/03-finetuning-tooling.md`](../.context/03-finetuning-tooling.md)).
- Qwen3-VL-4B LoRA is documented at ~2×21 GB — **does not fit a single T4**.

**Open questions to close first**
- Final backbone (shared with Step 3) and the train/val split over the 3,173 clips.
- Output format: bare class, or JSON with an explanation? The explanation feeds a bonus but costs
  tokens and latency.
- Window length, sampling rate, and the token-dial sweep — which axes, which values.
- Where training runs, and what the checkpoint/rollback discipline is.

**Depends on:** Step 1. Backbone decision shared with Step 3.
