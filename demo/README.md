# demo — run it

```sh
# 1. build the pages from a run's output (writes index.html and present.html)
PYTHONPATH=src .venv/bin/python demo/build.py \
    --events out/d1.events.jsonl out/d23.events.jsonl --name "qwen3-vl-4b zero-shot"

# 2. serve them
.venv/bin/python demo/server.py            # http://127.0.0.1:8800
```

| URL | What it is |
|---|---|
| `/` | **The presenter demo.** Full-screen, scripted, runs itself. This is the one to show an audience. |
| `/review` | The analyst view — all 34 videos, every prediction, scored. |
| `/slides` | The two submission slides. |
| `/health` | `{"ok": true, ...}` — check before you present. |

`--port 9000` and `--host 0.0.0.0` if you need them.

## Why a server rather than opening the file

`python -m http.server` **cannot serve video seeking** — it ignores HTTP `Range`, so the browser
asks for a slice, gets the whole file, and the player hangs. `demo/server.py` answers `206 Partial
Content` properly. Verified with a real media client:

```
ffmpeg -ss 400 -i http://127.0.0.1:8800/dataset/test/videos/T033.mp4 -frames:v 1 -f null -
```

decodes a frame 400 s into a 100 MB file over HTTP. It also answers `HEAD`, which Chrome's media
loader probes with before it will start a video — returning 405 there leaves the player stuck in
`NETWORK_LOADING` with no error to show for it.

The server serves the repo directory, so the pages' relative `../dataset/test/videos/...` paths
resolve identically whether they are served or opened straight off disk.

## The presenter demo

Seven scenes, chosen to tell the truth about the system rather than only its wins:

| | Video | What it shows |
|---|---|---|
| 1 | — | Title: the claim, and the measured cost |
| 2 | T009 | It works. Congestion at 0.98, with the model's own reasoning |
| 3 | T024 | A person lies on the pavement → loitering. Needs context, not just appearance |
| 4 | T012 | It reports fire **and** smoke. Both are there; the key credits only fire, so one scores as a false alarm |
| 5 | T004 | Ground truth is empty. The model reports congestion at 0.95. This is the expensive failure |
| 6 | T027 | Four minutes, four congestion events — it must also say *when*. One of four |
| 7 | T033 | Ten minutes, two real collisions, twelve alerts, neither caught |
| 8 | — | Scoreboard |

Alerts appear on the video's own clock: each one is raised when its window closes plus the run's
**measured** mean inference time, so the feed fills in the order a live operator would have seen it.

**It is a replay of recorded inference, and the page says so on screen.** The timings are real
measurements; the GPU is not running during the demo.

### Controls
`→` next · `←` back · `space` play/pause · `R` replay the scene · `F` full screen ·
click the dots to jump.

Video is muted so scenes autoplay — Chrome blocks autoplay with sound.

If a video cannot load, the scene says so on screen rather than showing a black rectangle.

## Rebuilding after a new run

Same `build.py` command with the new `--events` files. Both pages share one payload, so the
presenter view and the analyst view can never disagree.

Matching is not reimplemented here: `build.py` imports `ahc_vad.scoring.match_events` and calls it
with the same policies the scorer uses, so the pages cannot disagree with the score either.

## Three states it refuses to blur

- **no row in the events file** → `not run`, excluded from the tallies
- **`"failed": "<error>"`** → the run broke on that video, excluded
- **`"events": []`** → the model genuinely said normal, and *is* scored

Counting a broken or unrun video as a correct normal would flatter every partial run.
