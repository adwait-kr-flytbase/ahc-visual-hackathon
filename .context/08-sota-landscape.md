# SOTA landscape — surveyed 2026-09-05

Provenance: **read on the web**, not measured. Complements `02-prior-art.md` (the four organiser
papers), which is ~11 months behind the field in places. Cross-check against
`07-platform-and-scoring.md` for what actually earns marks.

---

## 1. The frontier moved: cascade-of-models → token-starved streaming VLM

`02-prior-art.md` concluded "cascade" is the consensus. That was the Oct-2025 answer. Through 2026
the efficiency work relocated **inside** a single always-on VLM, to the token budget:

| Work | Claim |
|---|---|
| StreamingVLM ([2510.09608](https://arxiv.org/abs/2510.09608)) | Infinite streams via compact KV cache — attention sinks + short vision window + long text window; training aligned to streaming inference |
| StreamingTOM (2510.18269) | Streaming token compression |
| StreamingAssistant (2512.12560) | Visual token pruning at **<1 ms**, latency independent of pruning ratio (A100) |
| **CodecSight (2604.06036)** | Prunes tokens using **H.264/265 codec motion vectors** mapped onto the ViT patch grid *before* encoding. Order-of-magnitude lower memory |
| ViCoStream (2606.19849) | Streaming VideoLLM **>100 FPS** via stage-wise coordinated inference |
| R3-Streaming (2605.17921) | Cascaded *agentic* control over streaming video, **95–96% fewer visual tokens** |
| Adaptive keyframe sampling (VideoBrain 2602.04094, Generative Frame Sampler) | **~8–10 pts accuracy** over uniform sampling |

**Reframe: cost is tokens, not parameters.** A 4B VLM on 32 frames costs more than a 9B on 4 well
chosen ones. Frame selection is simultaneously the latency dial *and* an accuracy lever — the one
knob that moves both the marks and the latency bonus.

## 2. Model shortlist has changed

- **Qwen3.5 small series** (released 2026-03-02): 0.8B / 2B / 4B / 9B dense, **natively multimodal
  via early fusion — there is no separate `-VL` variant**, 262K ctx, **Apache 2.0**, base
  checkpoints published. MMMU-Pro: **4B = 65.4%** vs Qwen3-VL-4B **52.0%**; 9B = 69.2% vs
  Qwen3-VL-8B 56.6%. 4-bit footprint: 4B ≈ 3 GB, 2B/0.8B < 2 GB. Described as the most capable
  multimodal models under 15B.
  → **Supersedes the Qwen3-VL-4B default in `03-finetuning-tooling.md`**, *and* it is what the
  AI City Challenge Track-3 winner used. **Verify video-input support before committing** — the
  writeups confirm image/vision, not explicitly video.
- **NVIDIA Cosmos 3** (2026-06): omnimodal physical-AI reasoning/world models; Nano + Super shipped,
  **Edge tier not yet released**. Tops the Traffic Anomaly Reasoning leaderboard. Ships **EVS
  (Efficient Video Sampling)** token reduction; runs on vLLM-omni + Dynamo.
- Sub-1B gate tier if a cheap always-on stage is wanted: Moondream 2B / 0.5B (int4/int8), SmolVLM.

## 3. The closest thing to this hackathon already ran: **AI City Challenge 2026, Track 3 — TAR**

Traffic Anomaly Reasoning. 3,670 transportation videos, 44,040 annotations, 10 task types with
explicit CoT traces; held-out TAR-Bench = 960 annotations over 80 clips. Official score averages
nine task types (**temporal localisation excluded** — the opposite of our weighting, where D2+D3
temporal IoU is 75 of 100 marks). Dataset is public:
`nvidia/PhysicalAI-Traffic-Anomaly-Reasoning` on HuggingFace → **directly usable extra training data**
(subject to the "may we train on outside data" question in `05-research-agenda.md` §15).

What won ([2608.17044](https://arxiv.org/abs/2608.17044)): teams moved *away from* prompting a
foundation model toward **agentic, evidence-first pipelines** — extract visual evidence with a video
captioner and an **open-vocabulary tracker**, then match that evidence to a task-specific answer
format consumed by an SFT'd VLM. Winner (Stellarview AI) used **Qwen 3.5**; TAU-Agent placed 2nd
(0.6779) with a retrieval agent orchestrating captioning + open-vocab tracking tools. Named winning
tactics: **metric matching** (shaping output to the scoring function) and **shared event memory**
reused across questions about the same video.

→ Confirms `02-prior-art.md`'s "describe-then-classify", and adds two things it missed:
**tracking belongs in the evidence layer**, and **fitting the metric is worth real points**.

## 4. What enterprises actually ship

- **NVIDIA Metropolis VSS blueprint** is the de-facto enterprise reference architecture: VLM
  (Cosmos) + LLM (Nemotron) + RAG + NIM microservices → *"real-time verified alerts"*, visual Q&A,
  automated reporting. Distributed via Accenture, Dell AI Factory, Lenovo; Lumana integrated it.
  The shipped pattern is **VLM → captions → vector store → LLM agent that decides whether to alert**,
  not a per-frame classifier.
- **Ambient.ai** runs an in-house VLM ("Pulsar"), edge-optimised, marketed on **90–95% false-alarm
  reduction**. Verkada / Coram / Lumana / Hakimo compete on the same axis.
  **The industry KPI is alarms-per-camera-per-day, not AUC** — which is exactly what the
  leaderboard's FA column punishes.
- Reported field reality: **0.9–5% false-alarm rates and 10–20% AUC drop on new sites.** Benchmark
  AUC does not transfer across camera domains.
- **FlytBase's own `AI-R`** runs VLMs on-site at the edge, no cloud, across DJI Dock 1/2/3, for
  real-time detection and automated response. This hackathon is scoping their product surface —
  worth reflecting in the two-slide deck. Percepto does onboard real-time analytics pushed to
  SCADA/ERP.

## 5. The ego-motion gap is real, and 2026 confirmed it

`02-prior-art.md` flagged that no supplied paper handles a moving camera. The moving-camera VAD
literature only *appeared* this year and is not VLM-era: **MUVAD** (a moving-UAV VAD benchmark,
built precisely because Drone-Anomaly and UIT-ADrone are hovering-only), **FTDMamba** (2601.11254),
dual-interval ego-motion decoupling (2605.22605), M²E-UAV (2605.10496).

**Idea worth testing (untested, mine):** replace Cerberus's frame-difference gate with **codec
motion vectors** (CodecSight's insight, repurposed) plus a **global-motion fit** — a homography or
affine fit over the MV field *is* the ego-motion; the **residual** is independently-moving objects.
Motion vectors fall out of the H.264/265 decoder for free, cost ≈0 GPU, and are ego-motion-robust by
construction. This is the missing piece that would make the Cerberus template work on drone and
dashcam footage, and it doubles as a **temporal proposal generator** for D2/D3 boundaries.

## 6. Other levers surfaced

- **RL post-training is now standard for temporal grounding**, not exotic: STVG-R1 (2602.11730),
  VTG-Reasoner, Video-R1 / T-GRPO, TAU-R1's GRPO, COPRA (2605.15325 — RL for VAD parameter
  adaptation). Given D2+D3 = 75 marks gated on **IoU ≥ 0.5**, SFT alone is known to underperform a
  GRPO stage with an IoU-shaped reward. **Too expensive for a 7 h build — stretch goal only.**
- **Track-state, not frame-state.** `stalled_or_broken_down_vehicle`, `loitering`,
  `traffic_congestion`, `vehicle_blocking_traffic` are *duration-defined*. Nothing in the reading
  list keeps per-object state. A cheap tracker plus per-track dwell time turns four of eleven classes
  into threshold logic and solves the alert de-duplication problem in `00-problem.md` §A. The AI City
  winners independently reached for an open-vocab tracker.
- **ASK-HINT remains the best training-free number**: 89.83% AUC on UCF-Crime with 6 prompts
  (Qwen2.5-VL-7B) — above fine-tuned Holmes-VAD (87.68) and HiProbe-VAD (88.91); weakly-supervised
  non-VLM SOTA (BN-WVAD) is 87.24. The training-free baseline is genuinely competitive, which
  matters when the realistic human ceiling on the board is ~50/100.

## 7. Strategic reading

Latency **is** scored, but as a **self-reported** `end_to_end_internal_time_ms` bonus
(`07-platform-and-scoring.md`), not measured by the host. So efficiency work pays twice — bonus
marks and the two-slide narrative — while accuracy marks are unconstrained by wall-clock. The clean
play is **one set of weights, two paths**: an accuracy-maximising offline path for the 100 marks,
and an instrumented streaming path that produces the honest latency numbers and the live demo.
§1's token-budget literature is what makes those two paths share a model instead of diverging.
