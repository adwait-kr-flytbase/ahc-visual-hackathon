# Two-slide deck — DRAFT

Required deliverable. Organisers: **exactly two slides**, covering *what you built · approach and
why · what you learned*, preferring **graphs / tables / timelines / example frames over paragraphs**,
and calling out anything that made it **faster, cheaper, more reliable, or lower false-alarm**.

`[[PLACEHOLDER]]` = waiting on a run. **No number goes in unmeasured.**

---

# SLIDE 1 — The data doesn't match the task. So we built the missing half.

**Title:** *Can a small VLM localise anomalies in long drone video? First you have to build the training data.*

### The gap we found (lead with this — it is the whole approach)

| | What we were given | What we are scored on |
|---|---|---|
| Videos | 3,173 **short** clips, 5–30 s | 240–629 s |
| Events | **exactly one** per clip | up to 6 per video, multiple classes |
| Timestamps | event ≡ whole clip for **7 of 11 classes** | boundaries gated at **tIoU ≥ 0.5** |
| Marks | D1 = **25** | D2 + D3 = **75** |

> **75 of the 100 marks had no training data at all.** Everything below follows from that.

### Pipeline

```
dataset/train ──┐
 3,173 clips    ├─► SYNTHESIZER ─► 360 long multi-event videos, exact ground truth
 973 normal ────┘   · filler from many clips  → cuts ≠ events
                    · random sub-spans        → duration ≠ class
                    · test-side res/fps       → closes the domain gap
                            │
                            ▼
                    windowed SFT rows ─► Qwen3-VL-4B + LoRA (frozen ViT)
                            │                    ▲ Modal A100
                            ▼
   video ─► 20 s windows ─► VLM ─► merge ─► events ─► submission JSON
                                    ▲
                        72 configs swept on CPU, ZERO re-inference
```

### Three choices that bought the most

| Choice | Why | Result |
|---|---|---|
| **Cache raw window predictions** | merge policy is worth as much as model quality when tolerance is 15 s | 72 configs tuned for **£0 of GPU** |
| **One shared prompt module** for training *and* inference | a one-token drift makes a fine-tuned model score **worse than zero-shot** | single source of truth, imported not forked |
| **Scorer built first, before any model** | the portal can't score synthetic data, and returns only 5 aggregate numbers | offline iteration in **seconds** |

---

# SLIDE 2 — We measured the shortcuts we created, and the bugs that don't crash.

### Results — 34 public videos

| Run | Model | Where | D1 /25 | D2 /35 | D3 /40 | FA | Runtime-legal? |
|---|---|---|---|---|---|---|---|
| All-empty baseline | — | — | `[[ ]]` | `[[ ]]` | `[[ ]]` | **0** | ✅ |
| Zero-shot | Qwen3-VL-4B | A100 | `[[ ]]` | `[[ ]]` | `[[ ]]` | `[[ ]]` | ✅ |
| Zero-shot | Cosmos-Reason2-8B | A100 self-hosted | `[[ ]]` | `[[ ]]` | `[[ ]]` | `[[ ]]` | ✅ |
| **Fine-tuned** | **Qwen3-VL-4B + LoRA** | A100 | `[[ ]]` | `[[ ]]` | `[[ ]]` | `[[ ]]` | ✅ |
| *Reference ceiling* | *Gemini 3.5 Flash* | *hosted API* | `[[ ]]` | `[[ ]]` | `[[ ]]` | `[[ ]]` | ❌ **dev-time only** |

*Frontier hosted models are a yardstick, not a submission — they cannot be in the runtime path.
And they are not a safe yardstick: **Gemini's 3rd video was a false positive at 0.85 confidence on
normal footage.***

### We manufactured training data — then measured the shortcut we'd created

Concatenating clips makes *event onset ≡ scene cut*. A model learns to detect **cuts**, scores
brilliantly on synthetic data, and collapses on real footage — which has no cuts.

| | before fix | after fix | target |
|---|---|---|---|
| P(event \| cut) | — | **0.10** | low ✅ |
| P(cut \| event start) | ~0.70 | **0.536** | 0.4–0.5 |

We censused pre-event footage per class first: **6 of 11 classes have literally none**
(loitering 0/300, waterlogging 0/95). So honest lead-in fixes 5 classes; the other 6 needed
interior jump cuts. **0.536 is near the honest floor — we report it rather than hide it, and we
trust real footage over synthetic tIoU.**

### Four silent failures. None of them crashed.

| # | Failure | Would have looked like |
|---|---|---|
| 1 | `ffmpeg -t` returns a **shorter** segment than asked, no error | every timestamp after it wrong |
| 2 | **49 of 2,092** train rows have `end ≤ start` | garbage spans, not an exception |
| 3 | Clip pool keyed on **folder**, not `class_name` — 108 relabelled rows | normal traffic taught as *wrong-way* |
| 4 | Empty model output ≡ correct "no events" | a parser bug scoring as a clean negative |

> **All four would have read as "the model just isn't very good."** We caught them by validating
> *intermediate artifacts*, not outputs. Every one is now a test.

### What we'd do next

**Every paper in the supplied reading list evaluates on fixed-camera CCTV. A third of this data is
drone, a third dashcam.** Frame-differencing gates assume a still background and break under
ego-motion. **Codec motion vectors** come free out of the H.264 decoder — fit a homography, and the
*residual* is independently-moving objects: an ego-motion-robust gate at ~zero GPU cost, which
doubles as a temporal proposal generator for D2/D3 boundaries.

---

## Notes for whoever renders this

- Slide 1 = **approach**, slide 2 = **evidence**. Don't split it any other way.
- The two tables that must survive any trim: *given vs scored* (slide 1) and *four silent failures*
  (slide 2). Those are the argument.
- Keep `REFERENCE CEILING` visually distinct from the runtime-legal rows. A judge misreading a
  Gemini number as our result is the worst outcome on the page.
- If a number is still `[[ ]]` at submission, **delete the row rather than guessing**.
