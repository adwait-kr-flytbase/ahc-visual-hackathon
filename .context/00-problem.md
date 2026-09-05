# Problem — distilled

> **Watch live drone video and emit actionable alerts for contextually-defined anomalies,
> using a model small enough to run in real time on limited GPU, with a false-alarm rate
> low enough that operators keep it switched on.**

**Event:** AHC Visual Intelligence Hackathon — Real-Time Video Anomaly Detection
**Date:** 05 September 2026 · FlytBase Labs / online
**Build window:** 11:00–18:00 (~7h). Demos 18:00–19:00.
**Morning session:** 09:30–11:00 — SOTA in video anomaly detection + robotics/CV related work + FlytBase demos.

---

## Input / output

| | |
|---|---|
| **Input** | Live drone video over a city — highways, streets, parks, railway stations, bus terminals, public gatherings, utility sites. **Day and night.** Overhead viewpoint, small objects, *moving camera*. |
| **Output** | An alert **while the drone is still overhead** — not a post-hoc review artifact. Something a responder on the ground can act on now. |

## Target events (explicitly NOT a fixed list)

1. Traffic congestion
2. Vehicle breakdown — incl. stopped on a highway, parked where it shouldn't be
3. A vehicle blocking traffic
4. Accidents
5. Smoke and fire
6. Water logging / flood
7. Loitering or suspicious presence
8. Road spill or debris
9. Fighting or violence
10. Wrong-way driving

Covering additional events that would matter to a ground responder is encouraged.
(Numbering in the source doc skips 7 — the list is 10 items, labelled 1–11.)

---

## The two tensions that define the problem

### 1. Anomaly is contextual, not object-level

> *"A stationary car is unremarkable in a parking bay and a problem on a highway shoulder."*

Closed-set detectors (YOLO) and classical CV answer *"what objects are here?"* Nothing about the
object identity determines anomaly status — same pixels, same class, same confidence, different verdict.
This is why the framing pushes toward VLMs: queryable in language, open-set, able to encode context
and cover events not specified in advance.

### 2. Capable VLMs are too expensive to run continuously

Large VLMs can do the reasoning but are too slow/costly for continuous live video across many feeds.

**→ The hackathon's actual research question: _can a small VLM do this reliably in real time?_**

---

## Constraints (hard)

- **Real time**, on **limited GPU capability** — economics must hold across many simultaneous drone feeds.
- Large hosted models are permitted **during development only**: comparison, distillation source,
  training-data generation. **They cannot be part of what makes the detector work at runtime.**

## Sanctioned solution directions (approach is otherwise open)

1. Fine-tune a small VLM on footage of this kind
2. Distil a larger model down
3. **Cascade** — lightweight always-on stage + heavier verification step
4. Train something purpose-built
5. Implement recent published work

---

## The two subtleties that separate a demo from a system

### A. Anomalies live on completely different time scales

| Event | Temporal shape |
|---|---|
| Accident | ~1 second, instantaneous |
| Congestion | builds gradually over minutes |
| Stopped vehicle | *only becomes* anomalous after being stationary for some time |
| Open drain / debris | not an event at all — a **static condition** |

Consequences a single-frame classifier cannot escape:
- "Stopped vehicle" is **defined by duration** → needs temporal state, not frame-in/label-out.
- A static condition (open drain) will be **re-alerted forever** by a per-frame model → needs
  alert de-duplication / event lifecycle, not just per-frame scoring.

### B. False positives are weighted equally with misses

> *"an alerting system that fires regularly on ordinary activity stops being used."*

This is a product requirement, not just a metric. Precision is first-class. Recall-maximising
demos that flag everything are explicitly called out as failures.

---

## What is provided

- **Unannotated** drone footage over urban areas, **including night flights**
- Public benchmark datasets, pre-downloaded (so setup doesn't eat build time)
- Sources span **drone, CCTV, and dashcam** footage
- Enough volume to fine-tune on
- Details in a separate **Dataset Doc** (shared separately — locate before the day)

⚠️ **Unannotated** — if you want supervised fine-tuning, label generation is part of the job.
This is exactly why large hosted models are explicitly permitted for training-data generation.

---

## Who it's for

ML/CV engineers · multimodal AI builders · researchers and students in video understanding,
VLM fine-tuning & distillation, efficient inference, anomaly detection, open-set recognition.

Participants are expected to arrive with a **tested** coding setup, model access, and a training
environment ready if they intend to fine-tune on the day.
