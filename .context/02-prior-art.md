# Prior art — read 2026-09-05

All four organiser-supplied papers retrieved and read. Distilled to what's usable here.

---

## ⭐ Cerberus — the template
*Real-Time VAD via Cascaded VLMs* · [arXiv 2510.16290](https://arxiv.org/abs/2510.16290)

**Closest paper to this hackathon's constraints. Treat as the reference architecture.**

Two-stage cascade, streaming:

| Stage | Model | Standalone speed |
|---|---|---|
| Coarse filter | `PE-Core-L14-336` CLIP + motion mask + rule deviation | ~118 fps |
| Fine reasoner | `Qwen2.5-VL-7B` captioner → `Qwen3-Embedding-4B` classifier | ~3.5 fps |

**The number that matters: 57.68 fps on one L40S at 1% anomaly rate, 97.2% AUC, 151.79× faster than VLM-only.**
End-to-end throughput scales with anomaly rate: 4.74 fps on raw benchmark data → 23.96 @5% → 57.68 @1%.
At realistic 1% anomaly, **only ~5% of frames reach the VLM**, at **≥95% anomaly recall**.

**Mechanisms worth stealing:**
- **Motion mask prompting** — frame-difference `p(D_t)=Σ|F_t−F_{t−1}|/(W·H)`; drop static frames below
  `ε=7e-4`; overlay **red circles** (subtle motion) or **red squares** (prominent, `p≥α=1.2e-3`) onto the
  frame to steer VLM attention. Cuts >50% of frames while holding 95%+ recall. Circles help fine accuracy,
  squares help coarse filtering; use both.
- **Rule-based deviation, not anomaly matching.** Offline: VLM captions normal segments → LLM (DeepSeek-R1)
  abstracts them into normality rules → augment with 339 Moments action labels as *perturbed* negatives.
  Online: health score `S(s)=Σ w_t·sim(v_s,v_t)` over top-k=5, `w=+1` normal / `−1` perturbed; anomalous if
  `S(s)<τ`. **Detects deviation by semantic competition** — no anomaly class list needed.
- **Fine→coarse feedback:** hard negatives from stage 2 are fed back into rule generation.
- **Precision/recall split by stage:** coarse stage sacrifices precision for recall; fine stage restores precision.

Evaluated on Avenue / SHTech / UBnormal / Campus — **all fixed-camera CCTV**, avg 97.24% AUC.
Ablation: removing action captioning costs −35.7% precision; cascade beats coarse-only by +14.9% AUC and
matches fine-only at 64× the speed.

> ⚠️ **Caveat for us:** Cerberus is built for a *static* camera — motion masking by frame differencing
> assumes a still background. On drone/dashcam footage ego-motion makes every pixel move. Needs
> stabilisation or a different cheap gate. This is the main adaptation cost.

---

## ASK-HINT — the cheapest experiment
*Unlocking VLMs for VAD via Fine-Grained Prompting* · WACV 2026 · [arXiv 2510.02155](https://arxiv.org/abs/2510.02155)

**Fully training-free**, frozen VLM. Structured prompting only.

Core claim: existing VAD prompts are **too abstract**, missing the human-object interactions and action
semantics that actually define an anomaly. ASK-HINT instead:
1. Organises prompts into **semantically coherent groups** (violence / property crimes / public safety)
2. Asks **fine-grained guiding questions** that tie predictions to discriminative visual cues

Beats both fine-tuned and training-free baselines on UCF-Crime and XD-Violence, generalises across VLM
backbones, and emits **interpretable reasoning traces**.

**Why it matters here:** zero training cost, so it's the natural day-one baseline and a fallback if
fine-tuning doesn't converge. The grouping idea maps cleanly onto our 12 classes — a traffic group
(`congestion`/`stalled`/`blocking`/`wrong_way`/`accident`), a hazard group (`fire`/`smoke`/`waterlogging`/
`spill`), a behaviour group (`fighting`/`loitering`). Reasoning traces are also demo-friendly.

---

## Alert-CLIP — the cheap stage, upgraded
*Abnormality-aware Latent-Enhanced Representation Tuning of CLIP* · CVPR 2026 · BUPT
[dataset](https://github.com/ClarkZhu216/Alert-CLIP_dataset)

**The finding, and it's a good one:** CLIP has *weak abnormality awareness* — normal and abnormal text
embeddings are **entangled**, so a video gets near-identical similarity to both prompts, and CLIP sometimes
scores the *wrong* description higher. This is a property of CLIP's representation space, not of prompt
wording — so no amount of prompt engineering fixes it. Their fix widens the normal/abnormal margin from
<0.16 to >0.38.

Method — OpenCLIP **ViT-L/32**, three alignment losses, two-stage curriculum:
1. **video–label** (global InfoNCE) — coarse semantic reshaping
2. **region–text** — ROI-Align on annotated anomaly boxes, aligned to region captions
3. **region–semantic** — contrast against hard negatives that look similar but flip normal/abnormal semantics

Frames → CLIP visual encoder → **lightweight temporal transformer** → clip-level embedding.
Trained on **VAGTA** (their re-annotation of UCF-Crime + MSAD: 4,212 clips, 3,726 train / 486 test, global +
region captions, 3 Qwen-VL-generated hard negatives per region). One A800 80GB.
Wins in **weakly supervised, zero-shot and open-vocabulary** settings; XD-Violence held out entirely.

**Why it matters here:** it's a **drop-in feature-extractor replacement** — the paper's own weakly-supervised
table is "swap the backbone in VadCLIP." If we build a Cerberus-style cascade, this is a better stage-1
encoder than vanilla CLIP. Bounding boxes are training-only supervision, not needed at inference.

---

## TAU-R1 — closest domain, weakest fit
*Traffic Anomaly Understanding* · [arXiv 2603.19098](https://arxiv.org/abs/2603.19098)

Also a **two-layer** design: lightweight classifier for coarse categorisation → larger reasoner for
detailed summaries. Same cascade instinct as Cerberus, independently arrived at.

Training: **decomposed-QA-enhanced SFT, then TAU-GRPO** (GRPO post-training with TAU-specific rewards).
Benchmark: **Roundabout-TAU** — 342 clips from real roundabouts (City of Carmel, Indiana), 2,000+ QA pairs.

**Least useful of the four.** Tiny benchmark, fixed roundabout CCTV viewpoint, no published latency numbers,
no drone/dashcam. Take the taxonomy framing and the SFT→GRPO recipe; skip the rest.

---

## What the four papers agree on

1. **Cascade.** Two of four independently converge on cheap-classifier → expensive-reasoner. This is the
   consensus answer to "VLM quality at real-time cost", and it's sanctioned direction #3.
2. **Frozen/lightly-tuned CLIP is the right cheap stage** — but vanilla CLIP's normal/abnormal margin is
   genuinely weak (Alert-CLIP), so tune it or expect a poor operating point.
3. **Prompt granularity is a real lever**, not a detail (ASK-HINT), and it costs nothing.
4. **Describe-then-classify beats classify-directly.** Cerberus and TAU-R1 both route through
   natural-language captions before deciding. Our `description_summary` column is exactly this signal.
5. **All four evaluate on fixed-camera CCTV.** UCF-Crime, XD-Violence, Avenue, SHTech, UBnormal,
   Roundabout-TAU — every one. **Nobody in this reading list handles a moving camera.**
   Our data is 1/3 drone + 1/3 dashcam. This is the open gap, and the likeliest place to win or lose.

## Still unread
- SOTA deck: [AHC_VAD_HACKATHON_SOTA.pptx](https://docs.google.com/presentation/d/1PiYW8hE5h8UNtveXxIxGm1U4q47h4_76/edit) — presented 09:30–11:00 on the day
