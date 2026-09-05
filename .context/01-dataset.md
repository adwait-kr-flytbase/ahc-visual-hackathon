# Dataset — Training and Public Test Data

Source: `sources/03-dataset.md`. **This doc materially changes the problem** — see
[§ Contradictions](#contradictions-with-the-problem-statement) at the bottom.

## Download
5 Google Drive mirrors (identical content). ~**15–17 GB** for the complete train+test pack.

- [Mirror 1](https://drive.google.com/drive/folders/1sEFKR7ctd5GfFw-nMlYd_MnTw1VVYz9K)
- [Mirror 2](https://drive.google.com/drive/folders/13E_CePn14lcbwMA_yZEiHpAVx6i09UIG)
- [Mirror 3](https://drive.google.com/drive/folders/13V8JqgZRMzn2TCF0HTsCqVgUH0UOMmpb)
- [Mirror 4](https://drive.google.com/drive/folders/1fS_i7QKXRDI6mnaI6UWqYzKSOYWG8rFv)
- [Mirror 5](https://drive.google.com/drive/folders/1efhUZhB6Kyvpw3RulZJSwd0brb8KhuZf)

## Layout

```
train/
  <class_name>/
    videos/*.mp4
    videos.csv
    ground_truth.csv
test/
  videos/*.mp4
  videos.csv
  ground_truth.csv
```

One folder **per final label, including `normal`**. Same two-file metadata pattern in every
train class folder and in the public test set.

## Label set — 12 classes, exact string match

```
normal                            traffic_accident
traffic_congestion                stalled_or_broken_down_vehicle
vehicle_blocking_traffic          wrong_way_driving
road_spill_or_debris              waterlogging_or_flood
fire                              smoke
fighting_or_violence              loitering_or_suspicious_presence
```

`class_name` must **match exactly**. This is now a **closed set** for scoring.

## `ground_truth.csv` schema — one row per event

| Column | Notes |
|---|---|
| `video_id` | **Repeats** — one video can hold several events |
| `level` | **1, 2 or 3** — the task tier |
| `is_anomaly` | Binary label |
| `class_name` | One of the twelve strings — match exactly |
| `start_time_sec` / `end_time_sec` | **Empty on Level 1, populated on Levels 2–3** |
| `description_summary` | Short natural-language description; **sometimes blank** |

`videos.csv` maps video IDs → files.
`normal` videos have **one row**, `class_name=normal`, empty event timestamps.

## Public test set
- **34 videos, ~56 minutes total**
- `ground_truth.csv` **included** — so the output format and scoring pipeline can be
  validated locally *before* submitting to the private evaluation system.

## Coverage
- **Viewpoints:** CCTV, dashcam, drone
- **Scenes:** highways, city streets, intersections, campuses, open areas
- **Conditions:** day, night, difficult weather / low visibility
- **Durations:** short isolated events **and** longer multi-event temporal videos

## Data separation
Train is separated from both the public test set and the **private evaluation set** at the
**original source-video / source-sequence level**. Different cuts of a reserved benchmark
source are never placed in training. → No leakage via near-duplicate clips.

## Provenance
Real samples selected from existing source datasets. **No synthetic anomaly footage.**
Videos standardized to a common delivery format, original visual events preserved.
Teams remain responsible for the usage terms of the underlying source datasets.

## Three intended usage modes (stated explicitly)
1. **Raw video** — build your own preprocessing / sampling pipeline
2. **+ `ground_truth.csv`** — anomaly, class and *temporal* supervision
3. **+ `description_summary`** — **vision-language fine-tuning or distillation**

Mode 3 is the organisers pointing directly at VLM SFT: the CSV already contains
video→text pairs in the shape a VLM wants.

---

## What this changes

### 1. There is a private evaluation system → this is a scored competition
Not an open-ended build with a demo. There's a submission format, a private test set, and a
score. **Getting the output format right is now a first-class task**, and the public test set
exists precisely so that can be de-risked early.

### 2. The data is annotated
The problem statement said *"unannotated drone footage."* It isn't. There is class supervision,
**temporal supervision** (start/end seconds), and **natural-language descriptions**.
Pseudo-labelling with Gemini/NIM is no longer the bottleneck it looked like.

### 3. The label set is closed and exact-match
The problem statement said the event list is *"not a fixed list"* and invited extra events.
The dataset says *"one of twelve strings — match exactly."* For anything that gets **scored**,
the taxonomy is fixed at 12. Extra-event coverage is a narrative/demo bonus, not a scoring axis.

### 4. Levels 1/2/3 formalise the temporal-scale problem
The tiering appears to be, judging by which fields are populated:

| Level | Fields | Likely task |
|---|---|---|
| 1 | timestamps **empty** | Video-level: is there an anomaly, and which class |
| 2 | timestamps **populated** | Temporal localisation: *when* did it happen |
| 3 | timestamps populated | Something further — multi-event? description? online/streaming? |

⚠️ **The doc says "the task tier (below)" but never defines the tiers.** Level 3 is undefined.
This is the single biggest open question and should be resolved from the CSVs
(see checks below) or by asking the organisers.

### 5. Multi-event videos are explicit
`video_id` repeats. A "longer multi-event temporal video" holds several events, possibly of
different classes, possibly overlapping. That rules out a pure single-label-per-video classifier
for Levels 2–3.

---

## Contradictions with the problem statement

| Topic | Problem statement | Dataset doc | Resolution |
|---|---|---|---|
| Annotation | "**Unannotated** drone footage" | Full `ground_truth.csv` with class + time + text | Dataset doc wins — but check whether extra *unannotated* raw drone footage ships alongside |
| Label list | "**not a fixed list**", ~10 events, extras welcome | **12 classes, exact match** | Closed set for scoring; extras are demo-only |
| fire / smoke | one event: "Smoke and fire" | **two separate classes** | Two classes |
| Open drain | given as a static-condition example | **not in the label set** | Not scored |
| `wrong_way_driving` | listed as "Wrong easy driving" (typo) | `wrong_way_driving` | Typo in the problem statement |
| Primary domain | framed as drone-first | CCTV + dashcam + drone, drone is one of three | Broader than drone-only |

---

## First checks to run once downloaded

```bash
# Class balance — how many events per class folder?
for d in train/*/; do echo -n "$d "; tail -n +2 "$d/ground_truth.csv" | wc -l; done

# What do the levels actually mean?
awk -F, 'NR>1{print $2}' train/*/ground_truth.csv | sort | uniq -c   # level distribution
# → per level: are timestamps populated? is description_summary populated?
# → does level correlate with class, or with video duration / multi-event-ness?

# Multi-event videos
awk -F, 'NR>1{print $1}' train/*/ground_truth.csv | sort | uniq -c | sort -rn | head

# Description coverage — how often is description_summary blank?
# (drives whether VLM SFT on descriptions is viable)

# Total duration per class, and the normal:anomaly ratio
# Video properties: resolution, fps, codec, duration distribution
# Viewpoint: is domain (cctv/dashcam/drone) recoverable from videos.csv or filenames?
```

*(Naive `awk -F,` will break on commas inside `description_summary` — use pandas/csv for anything
touching that column.)*

**Priority questions the data itself should answer:**
1. What distinguishes level 1 vs 2 vs 3? (undefined in the doc)
2. How much `normal` footage is there relative to anomalies?
3. What fraction of rows have a non-blank `description_summary`?
4. Is camera domain (CCTV / dashcam / drone) labelled anywhere, or must it be inferred?
5. What's the class balance — which of the 12 are rare?
6. Event duration distribution per class — does it match the instant/gradual/persistent split?

---

## Train split — **measured** 2026-09-05 12:00 IST (`dataset/train/`)

15 GB, 12 folders, **3,173 videos = 3,173 ground-truth rows (one event per video, never more)**.

| Class | Videos | Median event dur | Max |
|---|---|---|---|
| `normal` | **973** | – | – |
| `traffic_accident` | 565 | 5.0 s | 12.6 |
| `loitering_or_suspicious_presence` | 300 | **30.0 s** | 30.0 |
| `traffic_congestion` | 268 | 5.3 s | 20.0 |
| `stalled_or_broken_down_vehicle` | 223 | 8.9 s | 18.4 |
| `wrong_way_driving` | 164 | 5.0 s | 30.0 |
| `road_spill_or_debris` | 151 | 2.7 s | 27.0 |
| `vehicle_blocking_traffic` | 148 | 5.0 s | 30.0 |
| `fighting_or_violence` | 124 | **29.0 s** | 30.0 |
| `waterlogging_or_flood` | 95 | 5.8 s | 5.8 |
| `smoke` | 85 | 5.8 s | 30.0 |
| `fire` | 77 | 5.8 s | 30.0 |

**Schema differs from the doc:** the train CSVs have **no `level` column** —
`video_id,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary`.

- **`description_summary` is 100% populated** (3,173/3,173). VLM SFT on descriptions is fully viable.
- All 2,200 anomaly rows have timestamps; **70.5% start at t=0** and the event usually spans most of
  the clip. Clips are short (≤30 s), single-event, pre-trimmed around the event.
- Video format: **1280×720 h264** throughout, but **fps varies wildly — 15/8 (1.875), 15, 25, 30**.
  → **Always sample by timestamp, never by frame index.**
- Test D2/D3 videos are 25–30 fps, 240–629 s; T031 is 800×410 (source resolution preserved).

### ⚠️ The structural mismatch that defines the build

**Train has no long videos and no multi-event videos. D2 + D3 = 75 of 100 marks are exactly that.**
The public D2/D3 videos are visibly **composed from short clips on regular time grids**
(T025: six accidents on an exact 40 s grid; T028: four 5 s accidents on a 60 s grid).

→ **The training data for 75% of the marks does not exist and must be manufactured** by
concatenating train clips + `normal` filler into 240 s / 300–630 s videos with exact ground truth.
Training on the raw short clips alone teaches "the event spans the whole clip", which is the
worst possible prior for temporal localisation.
