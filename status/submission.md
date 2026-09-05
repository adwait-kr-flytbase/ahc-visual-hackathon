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
Public or host-accessible. **Nothing is committed yet.** This is the single biggest unforced risk on
the board right now: one `rm` and a day's work is gone, and the repo itself is graded.

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
A live demo is the obvious candidate if the clock allows.

---

## Checklist

- [ ] Upload the all-empty submission (floor ~13.5, zero risk) — **confirm this is done**
- [ ] Commit and push the repo ← **blocked on user approval**
- [ ] Fill in the portal profile (the leaderboard shows your real name)
- [ ] Submit a real zero-shot run
- [ ] Submit a fine-tuned run
- [ ] Write the two slides


## Two-slide deck — DRAFT ready

`status/slides.md` holds the full draft. Structure: **slide 1 = approach, slide 2 = evidence.**

Numbers are `[[PLACEHOLDER]]` until runs land. **Rule: if a number is still a placeholder at
submission time, delete the row rather than guess.**

The argument, in one line each:
1. The data does not match the task — 75 of 100 marks had no training data, so we built it.
2. We measured the shortcut we created (P(cut | event start) 0.70 -> 0.536) instead of assuming it away.
3. Four silent failures caught by validating intermediate artifacts, not outputs. None crashed.
4. Next: every paper in the reading list is fixed-camera CCTV; two thirds of this data moves.

Open question for the humans: **who renders it, and to what format?** The draft is markdown; it
still needs to become two actual slides. That is a 20-minute job someone has to own before 18:00.
