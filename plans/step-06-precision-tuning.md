# Step 6 — Precision and Threshold Tuning

> ## ✅ STATUS: `BUILT` — implemented in `src/vad/`, do NOT re-implement
>
> This brief was written before the inference session built the code. It exists now and is
> CPU-tested. **Adopt it and add tests, or bin it — but do not build both.**
> Several "open questions" below are already answered by working code; they are kept for the
> record. See [`../HANDOFF.md`](../HANDOFF.md).
>
> Modules: `vad/sweep.py` — re-merges cached window predictions over ~72 configs and scores each, no GPU, no re-inference

**Goal:** Decide how confident the system must be before it speaks. On this leaderboard that is
worth more than better detection.

**Why it exists:** an entrant with the second-best D1 recall scored **8.9/35** at D2 because of
**94 false alarms**. Guessing less would have scored more. See
[`../.context/07-platform-and-scoring.md`](../.context/07-platform-and-scoring.md).

**Data used**
- **Held-out synthetic long videos from Step 2** — the tuning set
- `dataset/test/` — sanity check only, **never the tuning target**

**The trap this step must not fall into.** Tuning on the 34 public videos is fitting to a set whose
answer key we already hold, where `stalled_or_broken_down_vehicle` and `wrong_way_driving` have
**one example each** and 16 of 52 rows are `traffic_accident`. "Suppress rare classes unless highly
confident" is optimal at n=1 and potentially catastrophic on a private set with a different mix.
Tune on synthetic, sanity-check on public, never the reverse.

**Rough shape:** per-class confidence thresholds rather than one global threshold; an event
lifecycle that merges near-duplicates and fires a persistent condition once rather than repeatedly.

**Open questions to close first**
- What confidence signal exists to threshold on? A generative VLM emitting JSON does not naturally
  produce a calibrated score — this may need logprobs, self-consistency over several samples, or a
  separate verifier pass. **Unresolved, and it gates the whole step.**
- Is suppressing rare classes ever the right call, given the private distribution is unknown?
- How do we detect that we are overfitting the tuning set?

**Depends on:** Steps 1, 2, 5.
