# demo — the live dashboard

An operations console: video on the left, the model's alerts on the right, the scoreboard along the
bottom, and a button that **runs the real model on an A100 while you watch**.

```sh
# 1. build the pages from a run's output
PYTHONPATH=src .venv/bin/python demo/build.py \
    --events out/d1.events.jsonl out/d23.events.jsonl --name "qwen3-vl-4b zero-shot"

# 2. deploy the live inference endpoint (once, close to demo time)
.venv/bin/modal deploy demo/modal_live.py        # prints the URL

# 3. put that URL in .env
MODAL_INFER_URL=https://<workspace>--ahc-demo-live-infer.modal.run

# 4. serve
.venv/bin/python demo/server.py                  # http://127.0.0.1:8800
```

| URL | What |
|---|---|
| `/` | **the dashboard** |
| `/review` | the analyst view — all 34 videos, every prediction scored |
| `/slides` | the two submission slides |
| `/api/backends` | which inference backends are live right now |
| `/health` | check before you present |

**Warm it up before you present.** The first call pays a ~50 s cold start while the model loads;
after that a window takes about 9 s. Press *Run* once before the audience is watching.

**Stop it afterwards** — `.venv/bin/modal app stop ahc-demo-live`. `min_containers` is 0 and the
container scales to zero 30 minutes after the last call, so it will not bill forever on its own,
but stopping it is the certain thing.

## What it shows

- **Live/recorded is always stated.** The pill in the header and the one on the video say which
  backend answered. Nothing on the page implies GPU work that did not happen.
- **Run the model** fires one 20 s window — the one under the playhead — through the real
  submission model. You get the classes, the confidences, the model's own sentence of reasoning,
  and a breakdown of where the time went: frame sampling vs model vs round trip.
- **Alerts** stream in on the video's clock from the recorded run, and live answers are added to the
  same feed with a `LIVE` badge, so the two are never confused.
- **Timeline** — the windows the model was shown, ground truth, and what it reported.
- **Scoreboard** — per difficulty, plus whole-run cost and a per-class bar where red is false alarm.

Keyboard: `space` play/pause · `←/→` 5 s · `Enter` runs the model.

## Backends, in order

1. **modal** — the submission model, Qwen3-VL-4B on an A100. Runtime-legal. Needs `MODAL_INFER_URL`.
2. **gemini** — a hosted reference model, needs `GEMINI_API_KEY`. Labelled in amber as **not**
   runtime-legal, because a hosted model cannot be in the submission's runtime path. It is also
   rate-limited on the free tier and will often refuse.
3. **none** — the dashboard falls back to the recorded run and says so.

Deadlines are per backend (`MODAL_DEADLINE_S`, default 120 s for the cold start; `GEMINI_DEADLINE_S`,
default 12 s) so a hanging backend fails fast and says why rather than freezing the button.

The Modal endpoint reuses `vad.frames.sample` and `vad.prompts` **verbatim** — the same 16 frames
over the same 20 s window as the scored runs. If that ever changes, the demo stops being the same
system and the page must say so.

## Why a server rather than opening the file

`python -m http.server` cannot serve video seeking — it ignores HTTP `Range`, so the player asks for
a slice, gets the whole file, and hangs. `demo/server.py` answers `206 Partial Content` properly.
Verified with a real media client:

```
ffmpeg -ss 400 -i http://127.0.0.1:8800/dataset/test/videos/T033.mp4 -frames:v 1 -f null -
```

It also answers `HEAD`, which Chrome's media loader probes with before it will start a video —
returning 405 there leaves the player stuck in `NETWORK_LOADING` with no error to show for it.

## Three states it refuses to blur

- **no row in the events file** → `not run`, excluded from the tallies
- **`"failed": "<error>"`** → the run broke on that video, excluded
- **`"events": []`** → the model genuinely said normal, and *is* scored
