# What we actually have to hand in

Two separate things. Both are graded. Only one is automated.

---

## 1 · The benchmark score — automated, unlimited attempts

One JSON file uploaded to the portal. **100 marks.**

| Difficulty | Videos | Task | Marks |
|---|---|---|---|
| D1 | 24 (T001–T024) | class only, no timestamps | 25 |
| D2 | 6 × 240s | class + start/end, 15s tolerance | 35 |
| D3 | 4 long (308–629s), multi-event | same, harder | 40 |

- D2/D3 events count **only** if the class is right **and** temporal IoU ≥ 0.5.
- **Best run stands.** A worse attempt never costs you → submit early, submit often.
- `events: []` means normal. Never emit `"normal"` as a class string. 11 classes, not 12.
- `explanation` (20–500 chars) earns a bonus and **can never reduce the score** → always send it.
- Latency bonus comes from **self-reported** `runtime_metadata`. Report it honestly.

**Status:** the all-empty baseline (~13.5 marks, validates the schema) is generated at
`out/submission-empty.json`. Confirm whether it has been uploaded — that's the floor and it costs
nothing.

Generating a real one is one command once a run finishes:
```
PYTHONPATH=src python -m vad.sweep  --windows out/<run>.windows.jsonl --gt dataset/test/ground_truth.csv \
    --manifest data/manifest.json --best-out out/<run>.best.jsonl
PYTHONPATH=src python -m vad.submit --events out/<run>.best.jsonl --manifest data/manifest.json \
    --out out/<run>.submission.json --model-name <name> --hardware "A100-40GB"
```

⚠️ **The public test set ships with its own answer key.** Tuning against it is self-deception — the
ranking that counts is the **private** eval set. That's why the other agent built ~100 synthetic
held-out videos: they're the only honest D2/D3 signal we have.

---

## 2 · The host-judged submission — manual, and worth a lot

### a. Code repository — REQUIRED
Public or host-accessible. **Committed and pushed** to
`github.com/adwait-kr-flytbase/ahc-visual-hackathon`, one agent owning all commits.
**Still private** — the spec requires the hosts can read it, so this has to be flipped before hand-in.

### b. Exactly two slides — REQUIRED, "carries high weight in the final judging"
Must cover: **what you built · approach and why · what you learned.**
Organisers explicitly prefer graphs, tables, timelines and example frames over paragraphs, and ask
you to call out anything that made it **faster, cheaper, more reliable, or lower false-alarm**.

**The story we can already tell, and it's a good one:**

1. **75 of the 100 marks had no training data.** Train ships only short single-event clips; D2/D3
   are long multi-event videos. We manufactured them — and measured the shortcut we'd created
   (P(event | cut) 0.10, P(cut | event start) 0.536) rather than assuming it away.
2. **Four silent failures caught by validating intermediate artifacts**, not outputs. Every one
   would have read as "the model isn't very good." (`experiments.md` has the list.)
3. **Cheap beats big where it counts.** Merge policy is swept over 72 configs on CPU with zero
   re-inference and, given a 15s tolerance, moves the score about as much as model quality does.
4. **The moving-camera gap.** Every paper in the reading list evaluates on fixed-camera CCTV; a
   third of this data is drone and a third dashcam. Codec motion vectors give an ego-motion-robust
   gate for free out of the decoder — the "what we'd do next" slide.

### c. Anything else — optional
**Built: `demo/index.html`.** Plays any test video with the ground truth and the model's predictions
on one time axis, and marks every span found / missed / false alarm live in the page. Rebuilds from
any run in one command:

```
PYTHONPATH=src python3 demo/build.py --run <name>
open demo/index.html
```

Why it is worth showing a judge rather than describing:
- **It scores itself against the answer key on screen.** A false alarm is a magenta band under an
  empty truth lane and a worded count next to the timecode. The system's worst failure mode is
  visible in one second, self-diagnosed, on real output.
- **A timing error is a shape.** Each matched truth event is joined to its prediction by a ribbon,
  so a perfect match is a rectangle and a drifted one is a visible skew. That is the D2/D3
  IoU >= 0.5 rule made legible without a formula.
- **It cannot disagree with our scorer.** `build.py` imports `ahc_vad.scoring.match_events`; there
  is no second matcher in `demo/`.
- **It refuses to flatter the run.** A video with no prediction reads `not run`, never "predicted
  normal", and is excluded from the tallies. Predictions the scorer rejects still appear, greyed,
  with the reason. This is what surfaced silent failure #7.

---

## Checklist

- [x] Commit and push the repo
- [ ] **Make the repo public or host-accessible** ← required, still private
- [ ] Upload the all-empty submission (floor ~13.5, zero risk) — **confirm this is done**
- [ ] Fill in the portal profile (the leaderboard shows your real name)
- [ ] Submit a real zero-shot run — blocked: the Gemini restart is writing failures as predictions
- [ ] Submit a fine-tuned run
- [x] **Two slides — DONE.** `slides/ahc-two-slides.pdf`, 2 pages, built from `slides/index.html`
- [x] Demo artifact (`demo/index.html`), verified on real output and on malformed input


## Two-slide deck — DRAFT ready

`status/slides.md` holds the full draft. Structure: **slide 1 = approach, slide 2 = evidence.**

Numbers are `[[PLACEHOLDER]]` until runs land. **Rule: if a number is still a placeholder at
submission time, delete the row rather than guess.**

The argument, in one line each:
1. The data does not match the task — 75 of 100 marks had no training data, so we built it.
2. We measured the shortcut we created (P(cut | event start) 0.70 -> 0.536) instead of assuming it away.
3. Four silent failures caught by validating intermediate artifacts, not outputs. None crashed.
4. Next: every paper in the reading list is fixed-camera CCTV; two thirds of this data moves.

**Rendered and done.** `slides/index.html` → `slides/ahc-two-slides.pdf` (2 pages, 1600×900).
Slide 1 is the approach — the data gap, the pipeline, measured real-time cost, model selection, and
a frame from the demo. Slide 2 is what we learned — the seven silent failures, our own retracted
headline, the two cheap policy wins, benchmark leadership failing to transfer, the motion-vector
negative result, and where the marks actually are.

Every number is measured. No fine-tuned result existed at build time, so none appears.
