# Step 2 — Long-Video Synthesizer

> ## ⚠️ STATUS: `DRAFT` — DO NOT IMPLEMENT
>
> This is a **brief**, not a spec. It records intent and the open questions, nothing more.
> Before any code is written, the open questions below must be closed with the user and this
> file rewritten to the standard of [`step-01-scoring-harness.md`](step-01-scoring-harness.md):
> exact files, exact interfaces, real test code, bite-sized TDD steps, no placeholders.
>
> **If you are an agent and you were told to "do step 2", stop and ask instead.**

**Goal:** Manufacture the training data for D2/D3 — long, multi-event videos with exact ground
truth — because `train/` contains none and D2+D3 is **75 of the 100 marks**.

**Why it exists:** measured in [`../.context/09-dataset-profile.md`](../.context/09-dataset-profile.md):
`train/` is 3,173 short clips, one event each, no `level`, no multi-event videos. The scored task is
event localisation inside 240–629 s footage. Nothing in the pack teaches that.

**Data used**
- `dataset/train/<class>/videos/*.mp4` — 2,200 anomaly clips, the event material
- `dataset/train/normal/videos/*.mp4` — 973 clips / 8.5 h, the filler and the negatives
- each folder's `ground_truth.csv` — where the event actually sits inside its clip

**Rough shape:** pick a target duration, lay down normal filler, splice in anomaly clips at chosen
offsets, record exactly where each went, re-encode to a uniform container. Output a video plus a
ground-truth CSV in the same schema as `dataset/test/ground_truth.csv`, so Step 1's scorer reads it
with no changes.

**Non-negotiable design constraints** (each derives from a measured fact — see the review section of
[`../.context/09-poc-plan.md`](../.context/09-poc-plan.md)):

1. **Events must not sit on the splice.** For 7 of 11 classes the labelled event *is* the whole
   clip, so naive concatenation makes event onset ≡ scene cut, and the model learns to detect cuts.
   Events need interior placement with lead-in/lead-out, and the filler needs distractor cuts at the
   same rate so a cut carries zero information.
2. **Event duration must vary independently of source-clip length.** `waterlogging` is 95 clips with
   5 distinct durations, all ≈5.7 s; `loitering` is 300 clips with **3** distinct durations. Copying
   clip lengths teaches duration→class. Real D3 durations span 2.6–125 s.
3. **Match the test-side format distribution, not the train-side one.** Train is 85.5% 1280×720 /
   15 fps; test D1 is 640×640, 720×404, 896×448, 256×192 at 24/30/1.875 fps. Randomise; letterbox,
   never stretch.
4. **Split source clips before synthesis.** Train and dev long-videos must be built from disjoint
   clip pools, or every dev number is leaked.
5. `ffmpeg`'s concat demuxer requires identical codec parameters, and train has 23 resolutions and
   10 frame rates — normalise each clip before concatenating or it fails silently.

**Open questions to close first**
- What mix of compositions, and how many videos? Mirroring the observed public-test patterns risks
  fitting to a synthetic quirk; the private set may be real continuous footage.
- How is "event position inside its source clip" recovered for the 7 classes where the annotation
  says the whole clip? Do we trust it, trim it, or detect the active span?
- Where does this run — local ffmpeg or Modal fan-out — and where does the output live?
- How much synthetic data is enough? Unknown until Step 4 gives a learning curve.

**Depends on:** Step 1 (its ground-truth schema and scorer).
