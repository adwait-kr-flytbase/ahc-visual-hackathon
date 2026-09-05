# Presentation prep — read this before you go up

**Score: 59.7 / 100.** D1 12.1/25 · D2 29.8/35 (**strongest tier**) · D3 14.2+/40
Started the day at 0. Was 49.9 an hour before the end.

---

## 1. What we built — say this first

> **A windowed vision-language pipeline that watches long video and reports anomalies with timestamps —
> plus a frozen-encoder classifier that catches what the VLM is blind to.**

```
video ──► sample frames BY TIMESTAMP ──► 20s sliding windows
             │                              │
             ▼                              ▼
      SigLIP2 + trained GRU head      Qwen3-VL-8B (prompted)
      "what is this, over time"       "what is happening here"
             │                              │
             └──────────► merge ◄───────────┘
                            │
                    per-difficulty policy
                            │
                     submission JSON
```

**The one-sentence version of why it works:**
**The VLM finds *appearance* — fire, smoke, flood, crashes. The frozen encoder finds *duration* — loitering, stalled vehicles.** Neither does both.

---

## 2. The three-agent system — how we actually worked

Three Claude Code sessions, on **one repository**, talking to each other by direct message.

| Agent | Role | Owned |
|---|---|---|
| **Agent 1** (coordinator) | Research, strategy, all GPU runs, all commits | `src/vad/`, `modal_app.py`, `status/` |
| **Agent 2** (builder) | Data pipeline, synthetic video generation, training, scoring code | `src/ahc_vad/`, `scripts/`, `tests/` |
| **Agent 3** (comms) | Demo, slides, process doc | `demo/`, `slides/`, `docs/` |

**The rule that made it work: ownership by directory.** No agent edits another's files. Cross-boundary changes are *requested*, not made.

**We learned that the hard way.** Early on, Agent 1 and Agent 2 both built the same four modules — scoring, submission, event types. We threw away Agent 1's copies. After that, one rule fixed it.

**One committer.** Agent 1 reviewed and pushed everything. The repo is a graded deliverable; three agents committing in parallel would have made the history unreadable.

**They corrected each other, out loud.** Real examples worth quoting:
- Agent 2 caught Agent 1 using **first-fit instead of best-IoU** matching in the scorer — silently wrong whenever predictions overlap.
- Agent 3 caught a **Gemini run that exited "successfully"** having done 6 of 34 videos, writing the 28 failures to disk as confident "normal" predictions.
- Agent 1 published *"bigger is worse, 4B beats 8B"*, then **retracted it** when asked for the mechanism — the models differ on 3 videos out of 24, p = 0.25. **That retraction was load-bearing: the 8B produced our best score.**

---

## 3. The knowledge factory — how three agents stayed coherent

Nothing important lived in chat. Everything went into files.

| File | What it is |
|---|---|
| `.context/` | **Shared memory.** Problem, dataset, prior art, decisions. Every finding written down, with provenance: *measured*, *read in a paper*, or *assumed*. |
| `status/ledger.md` | **The experiment ledger.** Every technique across 7 tiers — input, prompting, decoding, frozen heads, fine-tuning, RL, cascades — marked ✅ done / 🔄 running / ⬜ planned / ⛔ rejected **with the reason**. |
| `status/models.md` | Why each model was chosen, what it was compared against, what we rejected before testing. |
| `status/experiments.md` | Every run and its verdict, including the failures. |
| `plans/` | Specs written before code. |

**Why it mattered:** an agent could crash, restart, and pick up from the files. That happened. Twice.

---

## 4. Models we tested — the headline table

**All five on an identical 24-video set, same prompt, same policy, Modal A100.**

| Model | Size | Precision | Recall | F1 | Verdict |
|---|---|---|---|---|---|
| **Qwen3-VL-8B-Instruct** | 8B | 0.62 | 0.40 | 0.48 | **Shipped — produced our D1 + D2** |
| Qwen3-VL-4B-Instruct | 4B | 0.77 | 0.50 | 0.61 | Ties the 8B (differ on 3 of 24) |
| Cosmos-Reason2-8B | 8B | 0.57 | 0.20 | 0.30 | ❌ Half the recall |
| Qwen3-VL-2B-Instruct | 2B | 0.30 | 0.15 | 0.20 | ❌ Too small |
| Qwen2.5-VL-7B-Instruct | 7B | 0.00 | 0.00 | 0.00 | ❌ Silent on 20 of 24 |
| **SigLIP2 + GRU head** | frozen | — | — | — | **Shipped — +2.4 marks** |

**Three findings worth saying out loud:**

1. **Benchmark leadership does not transfer.** Cosmos-Reason2-8B is **purpose-built for traffic anomaly reasoning** and got *half the recall of a general 4B* on our task.
2. **Qwen2.5-VL-7B is ASK-HINT's own published backbone** (89.83% AUC on UCF-Crime in the paper). Here it returned a bare `[]` on 20 of 24 videos. Instruction-following failure, not blindness.
3. **We nearly didn't test SigLIP.** It was marked ⛔ *rejected* in our ledger on the theory that a frozen probe can't use context. It became one of the two things that actually moved our score. **We had rejected it on theory, without testing it.**

**Also tried and rejected before testing:** hosted models (banned from the runtime path), 72B/90B (too large), Llama-3.2-Vision (accepts **one image per request**; we send 8–32 frames).

---

## 5. Platforms

| Platform | Used for | Outcome |
|---|---|---|
| **Modal** | Every GPU job — A100-40GB and A100-80GB | ✅ The workhorse. ~$12 of $30 credit. |
| **Google Gemini API** | Reference ceiling, dev-time only | ⚠️ Works, but no trustworthy number — two runs killed by our own bugs |
| **NVIDIA NIM** | Hosted VLM catalogue | ❌ 4 of 5 models 404 — account entitlement, takes days |
| **HuggingFace** | Model weights, incl. gated Cosmos | ✅ |
| Kaggle | Free T4 fallback | Authenticated, never needed |
| Lightning AI | Considered | Not used — Modal covered it |

**Why Modal:** one Python file, `@app.function(gpu="A100")`, jobs run headless and detached. Persistent volumes held the dataset so nothing re-uploaded. **I could drive it entirely from the terminal**, which is what made three agents on one GPU budget workable.

---

## 6. The approach — and the two things that actually moved the score

**+9.8 marks came in the final hour. Fourteen VLM interventions before that moved nothing.**

### The big one: span geometry, +6.1
Eight prediction sets — three model sizes, five window configs, a fine-tune, targeted prompts — **all scored exactly 8.0 on D3.**

That invariance was the clue. **If changing the model changes nothing, the failure isn't in the model.**

Our D3 spans were **1–8 seconds inside videos of 327–602 seconds**. A 5-second span can never reach 50% overlap with a 100-second event — *geometrically impossible, no matter how right the class is.*

We widened spans to 112–138s. **49.9 → 56.0.** No new inference. No changed class. **Our classes had been right all along.**

> The evidence was in our own notes from that morning: the practice data had a **125-second** congestion event. We wrote down that D3 events are long — then spent hours swapping models instead of spans.

### The second: SigLIP2, +2.4
Frozen SigLIP2 → 8 frames → bidirectional GRU head → 12 classes.
It predicts **`loitering`** on two videos — a class we score **0/6** on and **no VLM proposed all day**.

---

## 7. What we'd say about the failures

**Ten silent failures in one day. None crashed.** Every one produced a plausible artifact.

Pick two or three of these to tell:
- **ffmpeg's `-ss` is a keyframe seek** — silently returns *more* than you ask for. A 468s video rendered as 241s and every timestamp after it was wrong.
- **The fix for one bug caused another.** We added per-video error handling so one bad video couldn't kill a 34-video run. It then converted a loud crash into **10 rows of confident, plausible, wrong output.**
- **SigLIP reported "validation accuracy 1.000".** Meaningless — 77 embeddings and a 7-sample validation split, because a 13GB upload had died unnoticed.
- **We deleted 125 synthetic videos we had already rendered** because their timestamps were silently wrong. *We threw away a third of our training data rather than train on labels we couldn't verify.*

**The line to land:**
> **In a seven-hour build, every failure that mattered was silent. So we validated intermediate artifacts, not outputs.**

---

## 8. Likely questions

**"Why not just fine-tune?"**
We did. LoRA on Qwen3-VL-4B, 300 steps, loss converged. It scored **42.4** replacing the baseline and **49.2** added to it — both **below** the 49.9 zero-shot. It trained on 2,610 rows instead of the full set because an upload failed, and 71% of its targets were empty, teaching it to stay silent when our problem was recall.

**"Is it real-time?"**
Measured, not asserted. **A window closes every 10 seconds and is scored in 1.83 seconds — 3.4× headroom, 2.8× at p95.** Self-reported latency, instrumented from the first call.

**"What would you do next?"**
1. **GRPO with a tIoU-shaped reward** — the literature says it beats SFT on temporal grounding, and 75 of the 100 marks are IoU-gated.
2. **Make SigLIP the cheap always-on stage** and the VLM the verifier. That's the cascade the brief actually asks for.
3. **Feed SigLIP the full dataset** — it earned +2.4 marks trained on a fifth of the data.

**"What are you still bad at?"**
Four classes we never detect: `loitering` 0/6, `fighting` 0/3, `road_spill` 0/3, `stalled` 0/2. All **context and duration** classes. The VLM is confident on appearance and blind to anything defined by time rather than pixels.
