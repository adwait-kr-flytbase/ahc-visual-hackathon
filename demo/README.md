# demo — the review page

One HTML file that plays a test video with the ground truth and the model's predictions on a
shared time axis, and marks every span found / missed / false alarm.

```sh
PYTHONPATH=src python3 demo/build.py --run gemini   # reads out/gemini.events.jsonl
open demo/index.html
```

`--events <path>` takes an explicit file instead of `--run`. `--out <path>` writes elsewhere;
keep the output inside `demo/` or the relative video paths break.

## What it shows

- **Two lanes on one axis.** Ground truth above, model below. The band between them draws a ribbon
  from each truth event to the prediction that matched it, so a timing error is a visible skew and
  a perfect match is a rectangle.
- **A miss** is an amber wedge that runs out before it reaches the model lane. **A false alarm** is
  a magenta wedge rising from nothing. Neither has a partner, and the shape says so.
- **Click any span** for the class, the span, IoU, the start/end error against its partner, and the
  model's own `explanation` string.
- **D1 has no timestamps in the ground truth**, so the truth bar is hatched across the whole clip
  and the ribbon band says timing is not scored. The page never implies a precision the data
  does not have.

## The two things it refuses to blur

- **"Not yet run" is not "predicted normal."** A video with no row in the events file is drawn as
  `not run` and excluded from the tallies. Counting it as a correct normal would inflate the score
  of every partial run.
- **A prediction the scorer rejects still appears**, greyed, with the reason — an unknown class or
  an inverted span. A demo that hides bad model output has the same defect as a scorer that does.

## Why it can't disagree with the scorer

`build.py` imports `ahc_vad.scoring.match_events` and calls it with the same policies
`ahc_vad.scoring.score` uses: class match only at D1, class match and temporal IoU ≥ 0.5 at D2/D3,
greedy best-IoU-first. There is no second implementation of matching in this directory, and the
page's JavaScript only draws what the build already decided.

## Notes

- Data is inlined into `index.html` rather than fetched: `file://` blocks XHR, and a venue demo
  should not need a local server or wifi. Videos stay relative at `../dataset/test/videos/`.
- Serving over `python3 -m http.server` works for the page but **seeking will hang** — that server
  ignores Range requests. Open the file directly.
