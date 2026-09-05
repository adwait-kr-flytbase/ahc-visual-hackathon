# Demo brief — run sheet

**http://127.0.0.1:8800** · an operations console: video left, alerts right, scoreboard bottom, and a
button that **runs the real model on an A100 while the audience watches**.

## Before you present — 3 minutes

```sh
.venv/bin/modal deploy demo/modal_live.py    # prints the URL
# put it in .env as MODAL_INFER_URL=...
.venv/bin/python demo/server.py              # http://127.0.0.1:8800
```

1. Open `/health` — confirms the backends are live.
2. **Press Run once, privately.** First call pays a **~50 s cold start** while the model loads.
   After that a window takes **~9 s**. Never let the audience watch the cold start.
3. Afterwards: `.venv/bin/modal app stop ahc-demo-live`.

## The 4 minutes on stage

| # | Do | Say |
|---|---|---|
| 1 | Play a short clip with a clean hit | "Frames sampled by timestamp, 20-second window, one call." |
| 2 | Point at the **timeline** | "Top row is what the model was shown. Middle is truth. Bottom is what it reported." |
| 3 | **Press Enter** on a window | "That's not a recording — that's an A100 answering right now." |
| 4 | Point at the **timing breakdown** | "1.8 seconds of model time against a 10-second window. 5.5× real time." |
| 5 | Show a **false alarm** (red) | "We show our misses. False alarms are the thing this problem punishes." |
| 6 | Scoreboard | "59.7 out of 100. D2 is our strongest tier." |

**Keyboard:** `space` play/pause · `←/→` 5 s · `Enter` runs the model.

## The honesty features — worth pointing out

- **Every alert says live or recorded.** A pill in the header names which backend answered. Nothing
  on the page implies GPU work that did not happen.
- **The live endpoint reuses `vad.frames.sample` and `vad.prompts` verbatim** — the same 16 frames
  over the same 20-second window as the scored runs. It is the same system, not a demo mock-up.
- **Three states are never blurred:** no prediction = *not run* · `"failed"` = the run broke ·
  `"events": []` = the model genuinely said normal, and *is* scored.
- Gemini is available as a fallback but labelled **amber, not runtime-legal** — a hosted model
  cannot be in the submission's runtime path.

## If it breaks

- **Button hangs** → per-backend deadlines fire (Modal 120 s, Gemini 12 s). It fails and says why.
- **Modal down** → falls back to the recorded run and states it on screen. The demo still works.
- **Video won't play** → you must use `demo/server.py`, not `python -m http.server`, which ignores
  HTTP Range and hangs the player.

## Other pages

`/review` — all 34 videos, every prediction scored · `/slides` — the two submission slides
