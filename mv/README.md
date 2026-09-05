# mv — codec motion vectors as an ego-motion-robust gate

**Verdict: the main hypothesis is falsified. One useful by-product survives.**

Every anomaly-detection paper in our reading list gates on frame differencing, which assumes a
static background. Two thirds of this dataset moves. The idea was that H.264 motion vectors —
already computed by the encoder, free out of the decoder, no GPU — could give a temporal proposal
generator that survives ego-motion: fit a global affine to the motion-vector field with RANSAC, and
whatever RANSAC rejects is moving independently of the camera.

```sh
.venv/bin/python mv/extract.py --stride 5 --workers 6   # -> mv/out/features.jsonl
.venv/bin/python mv/analyse.py                          # -> mv/out/report.json
.venv/bin/python mv/nulltest.py                         # permutation null
.venv/bin/python mv/plot.py                             # -> mv/out/*.png
```

## What was measured

All 34 test videos, 3391 s of footage, 75,242 frames decoded and 14,845 analysed, in **1165 CPU
seconds across 6 workers — 0.34× realtime, no GPU.** The extraction itself is cheap and that part
of the premise held.

Only D2/D3 can answer the question: D1 ground truth has no timestamps. That leaves 8 timed videos
with events, plus 2 timed normal videos as negatives.

## 1. Does residual motion mark the events? No.

AUC of "does this second fall inside a ground-truth event", scored by residual motion energy.
Threshold-free; chance is 0.500.

| feature | mean AUC (n=8) |
|---|---|
| residual energy | **0.556** |
| outlier fraction | 0.573 |
| spatial spread | 0.488 |
| blob count | 0.553 |

Per video it ranges from 0.384 to 0.877 — the spread is wider than the effect. `mv/out/residual-vs-events.png`
shows why: on T031 the trace is flat across the event boundary at 235 s and the only real spike sits
90 s inside the event; on T033 the two largest spikes in the video are both outside any event.

## 2. Do spikes mark event starts? Barely, and only at one threshold.

Spike = robust z above threshold, runs collapsed into one proposal, ±15 s tolerance to match the
portal's D2 boundary rule. Compared against a permutation null that shuffles proposal times within
each video, keeping the count per video fixed, 2000 trials.

| z | proposals | precision | null | p | recall | null | p |
|---|---|---|---|---|---|---|---|
| 2.0 | 127 | 0.346 | 0.298 | 0.077 | 0.846 | 0.765 | 0.190 |
| 2.5 | 91 | 0.330 | 0.294 | 0.209 | 0.692 | 0.618 | 0.249 |
| 3.0 | 72 | **0.417** | 0.309 | **0.010** | 0.692 | 0.544 | 0.051 |

Only z=3.0 separates from chance, at a 1.35× lift. That is one significant cell out of six tested;
Bonferroni-corrected it is p≈0.06. **Not a proposal generator.** 72 proposals to cover 26 events,
at a precision a coin-flip-and-a-third beats, would add false alarms to a system whose dominant
failure mode is already false alarms.

## 3. What did survive: ego-motion comes free, and the manifest doesn't have it

`manifest.json` ships `domain: ""` for all 34 videos — CCTV vs dashcam vs drone is never given.
The RANSAC fit's translation at the frame centre recovers it directly, per second, at no extra cost:

- **15 videos barely move** (<5% of seconds), **6 move part of the time**, **13 move throughout**.
- The separation spans four orders of magnitude, from 1.8e-6 (T010) to 3.8e-2 (T023).

It is a **continuum, not two classes** — there is no natural gap, and the 5e-4 line in
`mv/out/ego-motion.png` is a chosen threshold, not a discovered boundary.

**Measure it per second, never per video.** T033 is a dashcam video whose *median* ego-motion is
1.4e-4, below any sensible "moving" line, because it is composed from several source clips and the
static stretches dominate the median. Its first 40 s read 2.8e-3. A per-video summary hides exactly
the segments the feature exists to find.

## 4. One suggestive number that is NOT a result

Splitting the AUC by camera motion runs in the direction the hypothesis predicted:

| | n | mean AUC |
|---|---|---|
| static camera | 6 | 0.497 |
| moving camera | 2 | 0.732 |

**n=2.** Two videos, T032 and T034. This is far too small to claim anything, and it is exactly the
kind of number that is tempting because it agrees with the hypothesis we started with. Recorded so
nobody rediscovers it and believes it. Testing it properly needs more moving-camera videos with
timestamps than the public set contains.

## What this rules out

Ego-compensated motion energy is not a usable event detector on this data, so the cheap always-on
gate in a Cerberus-style cascade cannot be built this way. That is a real constraint on the
architecture, established in about an hour of CPU time, and it is worth more than a plausible
diagram of a gate nobody measured.
