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
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
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
    return RedirectResponse("/demo/present.html")


@app.get("/review", include_in_schema=False)
def review():
    return RedirectResponse("/demo/index.html")


@app.get("/slides", include_in_schema=False)
def slides():
    return RedirectResponse("/slides/index.html")


@app.get("/health", include_in_schema=False)
def health():
    built = (ROOT / "demo/present.html").exists()
    videos = len(list((ROOT / "dataset/test/videos").glob("*.mp4")))
    return {"ok": built and videos > 0, "present_built": built, "videos": videos}


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()

    if not (ROOT / "demo/present.html").exists():
        parser.error("demo/present.html is missing — run demo/build.py first")

    import uvicorn
    print(f"\n  demo    http://{args.host}:{args.port}/")
    print(f"  review  http://{args.host}:{args.port}/review")
    print(f"  slides  http://{args.host}:{args.port}/slides\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
