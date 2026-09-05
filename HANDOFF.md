# Two-lane build — contract and status

**Lane A (other agent):** GPU provisioning · long-video synthesizer · SFT.
**Lane B (this code):** everything from "a model exists" to "a scored submission JSON".

Lane B is **built and tested on CPU**. It runs today against zero-shot output and will accept the
fine-tuned adapter with a single flag change.

---

## Overlap with `src/ahc_vad/` — resolved on my side only

`src/ahc_vad/` (Lane A, step-01) is **canonical for data types, IO, submission and scoring**.
`src/vad/` (Lane B) is inference only. Where the two duplicated, I deleted my copy and imported
theirs. **No file under `src/ahc_vad/` was modified.** Their suite still passes (59 tests).

| Was duplicated | Canonical | Now |
|---|---|---|
| `vad.prompts.CLASSES` | `ahc_vad.taxonomy.ANOMALY_CLASSES` | imported; `vad.prompts` keeps an **ordered** list (prompts and merge need stable order) and asserts it matches the set |
| `vad.score._iou` | `ahc_vad.events.temporal_iou` | imported |
| `vad.score.load_gt` | `ahc_vad.groundtruth.load_ground_truth` | imported |
| my own submission writer | `ahc_vad.submission.build/validate/write` | **deleted**; `vad.submit` is now a thin adapter that maps jsonl → `Event` + `RuntimeMetadata`, then runs their validator before writing |
| my all-normal JSON | `scripts/make_empty_submission.py` | deleted mine |
| manifest parsing in `vad.run` | `ahc_vad.groundtruth.load_manifest` | still local — `vad.run` also needs to probe durations for synthetic videos with no manifest |

**One live duplicate, by necessity:** `src/vad/score.py` is a stopgap matcher. `ahc_vad/scoring.py`
is specified in step-01 but did not exist when `sweep.py` needed one. **When it lands, delete
`src/vad/score.py`** and point `sweep.py` at it — same signature
`score(pred_jsonl, gt_csv, manifest) -> dict`.

**Steps 3, 5, 6 and 7 are marked `DRAFT` in `plans/` — they are already built in `src/vad/`.**
Zero-shot baseline (3), long-video windowing and merge (5), threshold sweeping (6) and runtime
instrumentation (7) all exist and are tested. Re-spec them against this code rather than from
scratch, or bin this code — but do not build both.

## The three interfaces between the lanes

### 1. `src/vad/prompts.py` — the shared prompt. Import it, never fork it.

**This is the single highest-risk integration point.** If the training template and the inference
template differ by one token, the fine-tuned model scores worse than zero-shot.

Build every SFT row with:

```python
from vad.prompts import build_sft_sample   # PYTHONPATH=src
row = build_sft_sample(video_path, events, window_sec)   # -> one ms-swift jsonl line
```

`events` is a list of `{"class_name", "start", "end", "explanation"}` with times **relative to the
start of that clip/window**, not absolute in the source video. `build_sft_sample` emits the
system prompt, the `<video>` user prompt and the canonical assistant JSON target.

For a `normal` clip pass `events=[]` — the target becomes `{"events": []}`.

**Windowing the synthetic long videos:** train on windows, not whole 240 s videos. Cut each
synthetic video into the same 20 s windows the inference lane will use
(`vad.windows.plan(duration, win=20, hop=10)`), re-clip the ground truth into each window, and emit
one SFT row per window. Windows with no event become `{"events": []}` — those are the negatives that
buy precision.

### 2. Adapter path

Tell me the LoRA output dir. Inference then needs only:

```bash
PYTHONPATH=src python -m vad.run --engine hf --model Qwen/Qwen3-VL-4B-Instruct --adapter output/checkpoint-XXX ...
```

### 3. Synthetic dev set

Hold out ~100 synthetic long videos in their own directory with a `ground_truth.csv` in the same
shape as `dataset/test/ground_truth.csv`
(`video_id,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary`) plus a
`manifest.json` (`{"videos":[{"video_id","level","domain","duration_sec"}]}`). That is the only D2/D3
evaluation signal that exists — the portal cannot score it.

---

## Lane B: what exists

```
src/vad/prompts.py   SHARED contract: 11 classes, ASK-HINT grouped prompt, SFT sample builder,
                 tolerant response parser (a malformed generation yields [], never an exception)
src/vad/frames.py    timestamp-based sampling (clip fps ranges 1.875 -> 30; never index frames)
vad/windows.py   <=30 s -> one window; longer -> 20 s window / 10 s hop
vad/engine.py    HFEngine (transformers, works now) | ServerEngine (any OpenAI-compatible
                 endpoint: `vllm serve`, NIM, Gemini) -- same interface, swap with one flag
vad/merge.py     window predictions -> events. min_conf / gap_tol / min_dur / pad, all sweepable
vad/runtime.py   timers -> model_runtimes[] with p50/p95/max (this is the latency bonus)
vad/run.py       orchestrator -> <prefix>.windows.jsonl + <prefix>.events.jsonl
vad/score.py     matcher (see above)
vad/sweep.py     re-merges the raw windows under a grid and scores each -- NO re-inference
vad/submit.py    events.jsonl -> portal JSON
```

**Why `run.py` writes two files:** `windows.jsonl` holds the raw per-window predictions. `sweep.py`
re-merges those under ~72 configs and scores each **without touching the GPU**. Tuning merge policy
is therefore free, and with a 15 s tolerance and IoU ≥ 0.5 it is worth as much as model quality.

### Commands

```bash
# zero-shot baseline (no training needed)
PYTHONPATH=src python -m vad.run --videos dataset/test/videos --manifest data/manifest.json \
    --engine hf --model Qwen/Qwen3-VL-4B-Instruct --out out/zeroshot

# same, after the SFT run
PYTHONPATH=src python -m vad.run ... --adapter output/checkpoint-XXX --out out/sft

# high-throughput: vllm serve <model>, then
PYTHONPATH=src python -m vad.run --engine server --model <model> --base-url http://localhost:8000/v1 --workers 16 ...

# free tuning, no GPU
PYTHONPATH=src python -m vad.sweep --windows out/sft.windows.jsonl --gt dataset/test/ground_truth.csv \
    --manifest data/manifest.json --best-out out/sft.best.jsonl

PYTHONPATH=src python -m vad.submit --events out/sft.best.jsonl --manifest data/manifest.json \
    --out out/submission.json --model-name qwen3vl4b-lora --hardware "L40S 48GB"
```

`--limit 3` on `run.py` for a fast smoke test. `pip install opencv-python-headless pillow requests`
(plus transformers/peft/torch for the `hf` engine).

---

## Defaults worth challenging, in priority order

1. **`--frames 16` over a 20 s window is 0.8 fps** — probably too coarse for a 5 s accident.
   Sweep 16 / 24 / 32 against the synthetic dev set. This is the binding constraint on D2/D3.
2. **`--win 20 --hop 10`.** Median train event durations range 2.7 s (`road_spill_or_debris`) to
   30 s (`loitering`, `fighting`). One window size cannot suit both.
3. **`--d1-topk 1`** in `submit.py`. D1 ground truth is exactly one event per video and false alarms
   are punished hard, so top-1 is the safe default. Verify against the matcher before changing it.
