# Step 7 — Instrumentation, Demo and the Two-Slide Deck

> ## ⚠️ STATUS: `DRAFT` — DO NOT IMPLEMENT
>
> This is a **brief**, not a spec. It records intent and the open questions, nothing more.
> Before any code is written, the open questions below must be closed with the user and this
> file rewritten to the standard of [`step-01-scoring-harness.md`](step-01-scoring-harness.md):
> exact files, exact interfaces, real test code, bite-sized TDD steps, no placeholders.
>
> **If you are an agent and you were told to "do step 7", stop and ask instead.**

**Goal:** Produce honest latency numbers for the self-reported bonus, a live demo, and the two-slide
deck — which the organisers say **carries high weight in the final judging**.

**Data used:** nothing new. A demo video is chosen from `dataset/test/` — a long multi-event one
(T026 has four different classes in 240 s) shows the system best.

**Rough shape:** an instrumented streaming path that fills `runtime_metadata` per video
(`frames_processed`, `chunks_processed`, `end_to_end_internal_time_ms`, and per-model
`call_count`/`p50`/`p95`/`max`), plus a player with alerts overlaid as they fire.

**Deliverables from [`../.context/07-platform-and-scoring.md`](../.context/07-platform-and-scoring.md) §Step 2**
1. A code repository, public or host-accessible — **required**
2. Exactly **two slides** — required, high weight. Cover: what we built · the approach and why ·
   what we learned. Prefer graphs, tables, timelines and example frames over paragraphs. Call out
   anything that made it faster, cheaper, more reliable, or lower-false-alarm.

**Open questions to close first**
- Does the demo run live on stage, or is it a recording? That decides how much hardening it needs.
- Which two or three numbers carry the story? Pick them before building charts.
- Is the latency path the same code as the leaderboard path, or a separate streaming path?
  Two paths means two things to keep correct.

**Depends on:** Step 5. Start the deck early — it is graded, and it is always the thing that gets cut.
