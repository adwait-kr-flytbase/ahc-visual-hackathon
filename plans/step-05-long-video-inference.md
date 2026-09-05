# Step 5 — Long-Video Inference and Event Merging (D2/D3)

> ## ✅ STATUS: `BUILT` — implemented in `src/vad/`, do NOT re-implement
>
> This brief was written before the inference session built the code. It exists now and is
> CPU-tested. **Adopt it and add tests, or bin it — but do not build both.**
> Several "open questions" below are already answered by working code; they are kept for the
> record. See [`../HANDOFF.md`](../HANDOFF.md).
>
> Modules: `vad/windows.py` (<=30 s single window; else 20 s/10 s hop), `vad/merge.py` (min_conf / gap_tol / min_dur / pad)

**Goal:** Turn a clip classifier into an event localiser: slide over a long video, decide per
window, and merge decisions into events with start and end times. **75 of the 100 marks.**

**Data used**
- Synthetic long videos from Step 2 — training and honest dev
- `dataset/test/videos/*.mp4` (T025–T034) — final measurement only
- Real D3 footage is the **only trustworthy boundary metric**; synthetic tIoU will read optimistically

**Rough shape:** window the video, classify each window, then a merge state machine unions
overlapping same-class windows into events, with hysteresis so a single noisy window neither opens
nor closes an event. Optionally restrict windows to regions proposed by a cheap always-on gate
(the cascade from [`../.context/02-prior-art.md`](../.context/02-prior-art.md)).

**Open questions to close first**
- Window length and hop. A 20 s/10 s grid quantises boundaries to 10 s — is that inside tIoU ≥ 0.5
  for a 5 s event? (It is not.) Short events may need a different path from sustained ones.
- **Is there a cheap gate at all in v1**, or does the VLM see every window? The motion-vector
  ego-motion idea is the interesting version but it is unbuilt and unproven.
- How are boundaries refined below window granularity?
- Merge rules: max gap to bridge, minimum event duration, how to handle two classes overlapping in
  time (T026 has four different classes in 240 s).
- Blocked on the **D2/D3 matching rule** from Step 1 Task 7 — tight vs generous spans is the
  opposite strategy depending on the answer.

**Depends on:** Steps 1, 2, 4. **This is the highest-value step in the build.**
