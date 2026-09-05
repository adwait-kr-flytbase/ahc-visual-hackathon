# Dataset profile — **measured** 2026-09-05

All numbers below are measured from the downloaded pack (`dataset/`, 15 GB) via `ffprobe` over all
**3,207** videos. Raw sweep: `.context/artifacts/videometa.tsv` (one row per video).

---

## Shape

| | Videos | Rows | Duration |
|---|---|---|---|
| **train** | 3,173 | 3,173 (**exactly one row per video**) | **17.44 h** |
| **test** | 34 | 52 | 56.5 min |

**`train/` has no `level` column and no multi-event videos** — one clip, one event, one row.
Levels and multi-event structure exist **only in test**. Anything trained naively on `train/`
learns clip-level classification, which is D1 only — **25 of 100 marks**.

### Train class balance

| class | videos | total | p50 dur | max |
|---|---:|---:|---:|---:|
| normal | **973** | 509 min | 19.9 s | 3599 s |
| traffic_accident | 565 | 84 min | **5.0 s** | 155 s |
| loitering_or_suspicious_presence | 300 | 149 min | 29.9 s | 29.9 s |
| traffic_congestion | 268 | 36 min | 5.4 s | 20 s |
| stalled_or_broken_down_vehicle | 223 | 70 min | 12.3 s | 271 s |
| wrong_way_driving | 164 | 40 min | 11.0 s | 210 s |
| road_spill_or_debris | 151 | 28 min | 7.2 s | 128 s |
| vehicle_blocking_traffic | 148 | 43 min | 11.6 s | 230 s |
| fighting_or_violence | 124 | 44 min | 30.0 s | 187 s |
| waterlogging_or_flood | 95 | 9 min | 5.7 s | 5.8 s |
| smoke | 85 | 20 min | 5.8 s | 195 s |
| fire | 77 | 15 min | 5.8 s | 58 s |

Anomaly:normal = 2,200 : 973. **Not** the <1% rarity assumed in `05-research-agenda.md` §13 —
anomalies are *over*represented in train, the opposite of deployment. Rarest classes (`fire` 77,
`smoke` 85, `waterlogging` 95) are also the ones with near-zero duration variance — see below.

`description_summary` is **100% populated in train** (3,173/3,173); 37/52 in test.
Quality varies wildly — most are one-line stubs ("A traffic collision occurs."), a minority are
detailed causal paragraphs. Both are usable for SFT but the stubs carry almost no signal.

---

## ⚠️ Metadata is a strong provenance leak

Videos were re-encoded (85.5% of train is 15 fps / 1280×720) but **container tags and odd
frame-rates survived and fingerprint the source corpora.**

### 1.875 fps — a perfect source marker
- **Train: 300 videos at 1.875 fps, all 1280×720 — 100% of them `loitering_or_suspicious_presence`,
  and 0 loitering videos outside this signature.**
- **Test: 6 videos at 1.875 fps / 896×448 — T021, T022 (`fighting_or_violence`), T023, T024, T032,
  T034 (`loitering_or_suspicious_presence`).**

→ On the practice set the signature narrows those 6 videos to **2 of 11 classes** with certainty.
It is a *source-corpus* marker (a surveillance/behaviour dataset), not a class marker.

⚠️ **These videos are frame-starved**: 1.875 fps means a 20 s clip has **38 frames total**, and
T032 (307.7 s) has **577**. You cannot sample more frames than exist — any pipeline assuming
≥1 fps silently degrades here.

### `EV录屏` screen-recorder tag
584 train + 12 test videos carry `title=EV录屏…` / a Chinese screen-recorder comment.
Concentrated by class: **`waterlogging_or_flood` 95/95 (100%)**, `smoke` 55/85, `fire` 51/77,
`fighting_or_violence` 44/124, `normal` 222/973. In test it marks the 640×640 / 24 fps group
(T001, T002, T005, T008, T009, T012–T017).

### `encoder` tag
`Lavf60.3.100` (120 videos) is dominated by `wrong_way_driving` (52) + `vehicle_blocking_traffic` (49).
`Lavf56.36.100` and no-tag groups skew to road/traffic classes. Weaker than the above but real.

> **How to use this:** as a **prior, not an answer**. It is measured on the *practice* pack, whose
> ground truth we already hold, so it earns nothing on the leaderboard. Its value is (a) telling us
> the corpus is a mosaic of ~5 source datasets, and (b) *if* the private set is built by the same
> pipeline, the same signatures likely survive. Treat as a tie-breaker feature, never as the model.

---

## 🚩 Train and test do not look alike

| | train | test |
|---|---|---|
| dominant resolution | **1280×720 (85.5%)**, 23 distinct | **640×640, 720×404, 896×448, 256×192** at D1; 1280×720 / 1920×1080 at D2 |
| dominant fps | **15 fps (75%)** | 24 / 30 / 25 / 29.97 / 1.875 |
| duration | p50 ≈ 20 s, single event | D1 5.7–26.1 s · **D2 all exactly 240 s** · D3 307–629 s |

**640×640 is a square crop that appears nowhere in train.** D1 test clips are mostly *not* the
15 fps/720p format the training set was standardised to. Any model that overfits to 720p/15 fps
input geometry will degrade on the test pack. Resize/letterbox handling is not a detail here.

## Test structure — confirms the difficulty tiers

| Level | Videos | Duration | Structure |
|---|---|---|---|
| **D1** | 24 (T001–T024) | 5.7–26.1 s, 278 s total | one event per video, 4 normal, no timestamps |
| **D2** | 6 (T025–T030) | **all exactly 240.0 s** | 18 events over T025–T028; **T029/T030 are normal** |
| **D3** | 4 (T031–T034) | 307.7 / 360 / 376.5 / 628.8 s | 8 events, irregular boundaries |

**D2 is synthetic.** T025 = six accidents on an exact 40 s grid (20–40, 60–80 … 220–240);
T028 = four accidents on an exact 60 s grid, each exactly 5 s. Descriptions say so outright
("First separated traffic accident", "Second Accident-Bench collision"). T026 stacks four
*different* classes in one 240 s video. **A periodicity prior would score on D2 and is worthless
on D3** — do not fit to it.

**D3 is real footage.** Irregular precise boundaries (29.3–66.9, 235–360), sustained events
(a 125 s congestion build; two dashcam collisions 75 s and 45 s), and recurring loitering.
This is the honest tier — and it is worth the most (**40 marks**).

**T029 / T030 are 240 s of 1920×1080 normal** — pure false-alarm traps, the only 1080p in the pack.

---

## Consequences for the build

1. **The training set does not resemble the task.** Train = short trimmed single-event clips.
   Scored task = temporal localisation in long multi-event footage (D2+D3 = **75 of 100 marks**).
   Training data for the thing that carries most of the score **has to be constructed** — e.g. by
   concatenating train clips with known boundaries into synthetic long videos, mirroring how D2 was
   built. This is the single biggest gap between what we were given and what we are scored on.
2. **`normal` at 973 videos / 509 min is the most valuable class in the pack** given how brutally
   the leaderboard punishes false alarms. It is also the only source of hard negatives.
3. **Frame budget must adapt.** `03-finetuning-tooling.md`'s 16-frame default over a 628.8 s D3
   video is one frame per 39 s — hopeless for IoU ≥ 0.5. Long videos need windowing, not
   whole-video sampling. Conversely 1.875 fps clips cannot supply 16 distinct frames at all.
4. **Duration is a leaky class prior in train** (`waterlogging` is 95 clips all ≈5.7 s;
   `loitering` is 300 clips all ≈29.9 s). A model given whole clips may learn duration, not content,
   and that signal is absent at test time. Train on fixed-length windows, not whole clips.

## Two more measured facts that constrain the build

### The "event" is the whole clip for most classes
Share of train rows where `end−start` equals the clip duration (±0.5 s):

| ev == clip | classes |
|---|---|
| **100%** | `loitering` (3 distinct durations total), `waterlogging_or_flood` |
| 77–94% | `fighting` 94% · `fire` 91% · `traffic_congestion` 90% · `smoke` 82% · `traffic_accident` 77% |
| **4–45%** | `wrong_way_driving` 4% · `vehicle_blocking_traffic` 7% · `road_spill_or_debris` 8% · `stalled` 45% |

→ For 7 of 11 classes **train carries no within-clip boundary supervision at all** — the label is
just "this clip is class X". Only 4 classes have genuine sub-spans.

### No audio anywhere
Every video checked is **video-only, no audio stream**. Rules out audio-based cues entirely
(XD-Violence-style fusion, audio onset for `fighting`/`fire`). Closed door — do not spend time here.

## Open / unverified
- Camera domain (CCTV / dashcam / drone) is **not labelled anywhere** — `manifest.json`'s `domain`
  field is empty for all 34 videos. Must be inferred if we want it.
- Source separation between train and the private set is claimed by the organisers but unverified;
  no perceptual-hash check has been run.
