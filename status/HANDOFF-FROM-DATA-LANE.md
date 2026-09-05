# FINAL handoff — data lane → inference lane

**Written 14:55 IST by session `ahc-visual-hackathon-2-26` (data/training lane).**
The session I was coordinating with (`ahc-visual-hackathon-2-86`, inference lane) **disappeared**
between 14:50 and 14:55. This file exists so nothing is lost with it.

---

## PUSH THESE TO THE MODAL VOLUME, THEN FIRE THE TRAINING RUN

| Local | Rows / size | Volume destination |
|---|---|---|
| `out/sft/train.jsonl` | 5,633 rows, 9.8 MB | `/vol/sft/train.jsonl` |
| `out/sft/val.jsonl` | 296 rows, 528 KB | `/vol/sft/val.jsonl` |
| `out/sft/windows/` | 2,757 mp4, **1.1 GB** | `/vol/sft/windows/` |
| `out/synth-dev/` | 82 videos + gt + manifest | `/vol/synth-dev/` |

`out/synth-train/` is **optional** — windows are already cut from it; only needed to re-window at a
different win/hop. **Tar `out/sft/windows/` rather than uploading 2,757 files individually.**

## The training command — run verbatim, no edits needed

```bash
FPS_MAX_FRAMES=24 VIDEO_MAX_TOKEN_NUM=256 VIDEO_MAX_PIXELS=200704 \
swift sft --model Qwen/Qwen3-VL-4B-Instruct \
  --dataset /vol/sft/train.jsonl --val_dataset /vol/sft/val.jsonl \
  --train_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --freeze_vit true --freeze_aligner true \
  --torch_dtype bfloat16 --learning_rate 1e-4 --warmup_ratio 0.05 \
  --num_train_epochs 1 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 --gradient_checkpointing true \
  --max_length 8192 --save_steps 200 --logging_steps 10 \
  --output_dir /vol/output/qwen3vl4b-lora
```

**Given the clock:** 5,633 rows at batch 1 / accum 8 is ~700 optimizer steps. If that exceeds ~90 min
on the A100, add `--max_steps 400` rather than reducing the epoch — shuffled data makes a partial
epoch fine, and `save_steps 200` yields a usable adapter at step 200 and 400.
**An adapter that exists at 17:00 beats a better one that finishes at 17:50.**

Departures from the organisers' snippet, all deliberate:
- `FPS_MAX_FRAMES` 16→24 — 16 over a 20 s window is 0.8 fps and cannot see a 5 s accident
- `VIDEO_MAX_TOKEN_NUM` 128→256, `max_length` 4096→8192 — 24 frames will not fit otherwise
- `lora_rank` 8→16 — ~5.6k rows is still cheap at 16

If it OOMs: drop `VIDEO_MAX_PIXELS` to 100352 first, then `FPS_MAX_FRAMES` to 16, then `max_length`
to 6144. **Return the traceback rather than silently changing dials.**

## Verified, not assumed

- 400 sampled video paths resolve under the `/vol` → repo mapping. **0 missing.**
- 2,724 event targets: all parse as JSON, all 11 classes valid, all satisfy `0 <= start < end`.
- System and user prompts **byte-identical** to `vad.prompts.SYSTEM` / `user_prompt()` — asserted in
  code, not eyeballed. This was the highest-risk integration point; it is closed.
- 52% of train rows are negatives (`{"events": []}`) — the precision budget.
- `synth-dev` self-scores **100.0** through `ahc_vad.scoring`.

Class balance across 2,724 events:
`traffic_accident` 572 · `congestion` 406 · `loitering` 340 · `stalled` 254 · `road_spill` 228 ·
`blocking` 182 · `fire` 176 · `smoke` 167 · `fighting` 162 · `waterlogging` 161 · `wrong_way` 76

`wrong_way` is thinnest at 76 — expected, it lost 108 rows to the organisers' relabelling.

## Uncommitted work in the data lane (I am not the committer)

| File | State |
|---|---|
| `src/ahc_vad/synth.py` | sustained-class chaining, lead-in/tail, `class_name` keying |
| `src/ahc_vad/tracks.py` | track-state layer — **designed, NEVER RUN** |
| `scripts/make_sft.py` | window downscaling to 448, `/vol` path prefix |
| `scripts/recover_synth_metadata.py` | NEW — rebuilds metadata after a killed render, ffprobe-gated |
| `scripts/run_tracks.py` | NEW — YOLO+ByteTrack runner, **never executed** |
| `status/slides.md` | two-slide draft |

**The track layer must never be reported as a result.** It was never run — the torch install was
OOM-killed twice. In the deck it is "designed, not measured". If an A100 slot frees up, `--device cuda`
should work, but do not quote a number that was not scored against `dataset/test`.

## Data caveats the next person needs

1. **`dataset/train/wrong_way_driving/` is MIXED** — 56 wrong-way + 108 relabelled `normal`.
   Key clips on `class_name`, never on folder name. A test guards this.
2. **46 of 2,092 localised train rows have `end <= start`.** `load_ground_truth` skips and counts them.
3. **ffmpeg `-ss` before `-i` is a keyframe seek** and returns MORE than requested. It drifted 4–8 s
   on 35 videos. `recover_synth_metadata.py` ffprobe-gates every file; anything re-encoding these
   clips has the same exposure.
4. Synthetic sets are smaller than planned — 82 dev / 153 train — because drifted videos were dropped.
   **153 correct videos beat 200 with silently wrong timestamps.**
