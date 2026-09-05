"""Serve the demo over HTTP, with the byte-range support video scrubbing needs.

`python -m http.server` ignores Range requests, so a browser cannot seek: the video element
asks for a slice, gets the whole file with a 200, and hangs. Chrome will not scrub a 100 MB
dashcam clip through that. This serves 206 Partial Content properly, which is the whole reason
this file exists rather than a one-liner.

    .venv/bin/python demo/server.py            # http://127.0.0.1:8000
    .venv/bin/python demo/server.py --port 9000 --host 0.0.0.0

  /            the presenter demo   (demo/present.html)
  /review      the analyst view     (demo/index.html)
  /slides      the two slides       (slides/index.html)

Serves the repo directory, so the pages' relative `../dataset/test/videos/...` paths resolve
here exactly as they do when the file is opened straight off disk.
"""

import argparse
import mimetypes
import os
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
CHUNK = 1024 * 1024
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

app = FastAPI(title="AHC anomaly demo", docs_url=None, redoc_url=None)


def safe_path(relative: str) -> Path:
    """Resolve inside ROOT or refuse. Keeps `..` in a URL from reaching the filesystem."""
    target = (ROOT / relative).resolve()
    if not target.is_relative_to(ROOT) or not target.is_file():
        raise HTTPException(404, f"no such file: {relative}")
    return target


def ranged(path: Path, request: Request) -> Response:
    """Serve a file, honouring Range so the browser can seek."""
    size = path.stat().st_size
    media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = request.headers.get("range")
    if not header:
        return FileResponse(path, media_type=media,
                            headers={"accept-ranges": "bytes", "cache-control": "no-cache"})

    match = RANGE_RE.fullmatch(header.strip())
    if not match:
        raise HTTPException(416, "malformed Range")
    raw_start, raw_end = match.groups()
    if raw_start:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    else:                                   # suffix form: bytes=-500 means the last 500 bytes
        if not raw_end:
            raise HTTPException(416, "malformed Range")
        start, end = max(0, size - int(raw_end)), size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        return Response(status_code=416, headers={"content-range": f"bytes */{size}"})

    def stream():
        remaining = end - start + 1
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                block = handle.read(min(CHUNK, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block

    return StreamingResponse(stream(), status_code=206, media_type=media, headers={
        "content-range": f"bytes {start}-{end}/{size}",
        "content-length": str(end - start + 1),
        "accept-ranges": "bytes",
        "cache-control": "no-cache",
    })


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/demo/dash.html")


@app.get("/review", include_in_schema=False)
def review():
    return RedirectResponse("/demo/index.html")


@app.get("/slides", include_in_schema=False)
def slides():
    return RedirectResponse("/slides/index.html")


# ---------------------------------------------------------------- live inference
# The demo runs the real thing when it can. Backends are tried in this order and the answer
# always names which one served it, so the dashboard can never imply GPU work that did not happen.
#   modal   the submission model on an A100. Set MODAL_INFER_URL.
#   gemini  a hosted reference model. Works today, but it is NOT runtime-legal for the submission.
#   none    nothing available; the dashboard falls back to the recorded run and says so.

# Per-backend deadlines. Modal may pay a ~50 s cold start on the first call of a demo; Gemini
# retries 429s with backoff for ~30 s and is only a fallback, so it gets cut off much sooner.
DEADLINES = {"modal": float(os.environ.get("MODAL_DEADLINE_S", "120")),
             "gemini": float(os.environ.get("GEMINI_DEADLINE_S", "12"))}


class InferRequest(BaseModel):
    video_id: str
    start: float
    end: float
    frames: int = 16
    backend: str | None = None          # force one, otherwise best available


def available_backends() -> list[str]:
    out = []
    if os.environ.get("MODAL_INFER_URL"):
        out.append("modal")
    if os.environ.get("GEMINI_API_KEY"):
        out.append("gemini")
    return out


def _run_modal(req: InferRequest) -> dict:
    import requests
    url = os.environ["MODAL_INFER_URL"].rstrip("/")
    headers = {}
    if os.environ.get("MODAL_INFER_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["MODAL_INFER_TOKEN"]
    started = time.time()
    r = requests.post(url, json=req.model_dump(exclude={"backend"}), headers=headers, timeout=120)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    payload.setdefault("ms", int((time.time() - started) * 1000))
    payload.setdefault("model", "Qwen/Qwen3-VL-4B-Instruct")
    return payload


def sample_fast(path: Path, t0: float, t1: float, n: int, max_side: int = 640):
    """Frames for one window, via ffmpeg input-seek.

    src/vad/frames.py seeks with cv2 CAP_PROP_POS_MSEC, which walks the file: 30 s to pull 16
    frames from a 100 MB clip, and this has to feel live. Putting -ss before -i lets ffmpeg jump
    on the index instead. Falls back to the pipeline's own sampler if ffmpeg is unavailable.
    """
    import io, subprocess
    from PIL import Image

    span = max(t1 - t0, 1e-3)
    rate = n / span
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(path), "-t", f"{span:.3f}",
           "-vf", f"fps={rate:.6f},scale='min({max_side},iw)':-2",
           "-frames:v", str(n), "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"]
    try:
        blob = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    except Exception:
        blob = b""
    frames = []
    start = blob.find(b"\xff\xd8")
    while start != -1:
        end = blob.find(b"\xff\xd9", start)
        if end == -1:
            break
        frames.append(Image.open(io.BytesIO(blob[start:end + 2])).convert("RGB"))
        start = blob.find(b"\xff\xd8", end)
    if frames:
        return frames
    from vad import frames as F
    return F.sample(str(path), t0, t1, n=n)[0]


def _run_gemini(req: InferRequest) -> dict:
    """Sample frames here and send them to Gemini, using the pipeline's own prompt and parser."""
    from vad import prompts as P
    from vad.engine import GeminiEngine

    path = ROOT / "dataset/test/videos" / f"{req.video_id}.mp4"
    if not path.exists():
        raise HTTPException(404, f"no video {req.video_id}")
    sampled = time.time()
    imgs = sample_fast(path, req.start, req.end, req.frames)
    sample_ms = int((time.time() - sampled) * 1000)
    if not imgs:
        raise RuntimeError(f"could not sample frames from {req.video_id} at {req.start:.0f}s")
    span = req.end - req.start
    engine = GeminiEngine(os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"))
    started = time.time()
    text = engine.generate(imgs, P.SYSTEM, P.user_prompt(span))
    ms = int((time.time() - started) * 1000)
    events = P.parse_response(text, span) or []
    for e in events:                       # window-relative -> absolute video time
        if e.get("start") is not None:
            e["start"] = round(req.start + e["start"], 2)
            e["end"] = round(req.start + e["end"], 2)
    return {"events": events, "ms": ms, "sample_ms": sample_ms, "model": engine.name,
            "frames": len(imgs), "raw": text[:400]}


@app.get("/api/backends")
def backends():
    return {"available": available_backends(),
            "preferred": (available_backends() or ["none"])[0]}


@app.post("/api/infer")
def infer(req: InferRequest):
    """Run one window through the model and report the measured latency."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

    order = [req.backend] if req.backend else available_backends()
    errors = {}
    for name in order:
        runner = {"modal": _run_modal, "gemini": _run_gemini}.get(name)
        if runner is None:
            continue
        # A hard deadline: GeminiEngine retries 429s with backoff for ~30 s, and a demo button
        # that hangs that long is worse than one that fails and says why. Not a `with` block --
        # ThreadPoolExecutor.__exit__ waits for the worker, which would undo the timeout.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            deadline = DEADLINES.get(name, 30.0)
            result = pool.submit(runner, req).result(timeout=deadline)
        except FutureTimeout:
            errors[name] = f"{name} did not answer within {deadline:.0f}s"
            continue
        except HTTPException:
            raise
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        result["backend"] = name
        result["live"] = True
        result["runtime_legal"] = name == "modal"
        result["window"] = [req.start, req.end]
        return result
    return {"events": [], "backend": "none", "live": False, "runtime_legal": False,
            "errors": errors, "window": [req.start, req.end],
            "detail": "no live backend answered; the dashboard shows the recorded run instead"}


@app.get("/health", include_in_schema=False)
def health():
    built = (ROOT / "demo/dash.html").exists()
    videos = len(list((ROOT / "dataset/test/videos").glob("*.mp4")))
    return {"ok": built and videos > 0, "dash_built": built, "videos": videos,
            "backends": available_backends()}


# HEAD as well as GET: Chrome's media loader probes with HEAD before it will start a video,
# and a 405 there leaves the element stuck in NETWORK_LOADING with no error to show for it.
@app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
def any_file(path: str, request: Request):
    target = safe_path(path)
    if request.method == "HEAD":
        return Response(status_code=200, headers={
            "content-length": str(target.stat().st_size),
            "content-type": mimetypes.guess_type(target.name)[0] or "application/octet-stream",
            "accept-ranges": "bytes",
        })
    return ranged(target, request)


def load_env() -> None:
    """Read .env so GEMINI_API_KEY / MODAL_INFER_URL are available without exporting them."""
    path = ROOT / '.env'
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()

    load_env()
    if not (ROOT / "demo/dash.html").exists():
        parser.error("demo/dash.html is missing — run demo/build.py first")

    import uvicorn
    print(f"\n  demo    http://{args.host}:{args.port}/")
    print(f"  review  http://{args.host}:{args.port}/review")
    print(f"  slides  http://{args.host}:{args.port}/slides\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
