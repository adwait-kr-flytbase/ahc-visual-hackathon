"""Live single-window inference for the on-stage demo.

One HTTP call = one 20-second window through the real submission model on an A100. Same sampling
and the same prompt as the scored runs, so what the audience sees is the system, not a lookalike.

    .venv/bin/modal deploy demo/modal_live.py     # prints the URL
    .venv/bin/modal app stop ahc-demo-live        # WHEN DONE — a warm A100 bills while deployed

Separate Modal app from modal_app.py so deploying the demo can never disturb the scored runs.
"""

import modal

app = modal.App("ahc-demo-live")
vol = modal.Volume.from_name("ahc-vad-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("ahc-vad-hf", create_if_missing=True)

# `fastapi[standard]` is required explicitly now — Modal used to inject it and no longer does.
# Without it `modal deploy` fails outright on any web endpoint.
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("opencv-python-headless", "pillow", "requests", "numpy", "fastapi[standard]")
    .pip_install("torch==2.6.0", "torchvision", "transformers>=4.57", "accelerate", "peft",
                 "qwen-vl-utils", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf", "PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")          # must be last: Modal forbids build steps after it
)

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
_ENGINE = None


@app.function(image=IMAGE, gpu="A100-40GB", volumes={"/vol": vol, "/hf": hf_cache},
              secrets=[modal.Secret.from_name("hf-token")],
              scaledown_window=1800, timeout=600)
@modal.fastapi_endpoint(method="POST", docs=True)
def infer(item: dict):
    """One window of real inference. Returns measured model time, never a configured number.

    min_containers is deliberately 0: a warm A100 bills for as long as it is deployed. The first
    call pays the cold start, so warm it once before presenting. scaledown_window keeps it up for
    30 minutes after that.
    """
    import glob
    import sys
    import time

    sys.path.insert(0, "/root/src")
    from vad import frames as F, prompts as P
    from vad.engine import HFEngine

    global _ENGINE
    video_id = str(item.get("video_id", ""))
    start = float(item.get("start", 0.0))
    end = float(item.get("end", start + 20.0))
    n_frames = int(item.get("frames", 16))
    model = item.get("model", MODEL)

    hits = (glob.glob(f"/vol/dataset/test/videos/{video_id}.mp4")
            + glob.glob(f"/vol/evalvideos/{video_id}.mp4")
            + glob.glob(f"/vol/dataset/train/*/videos/{video_id}.mp4"))
    if not hits:
        return {"error": f"video {video_id} is not on the volume"}

    try:
        cold = _ENGINE is None or _ENGINE.name != model
        load_ms = 0
        if cold:
            loading = time.perf_counter()
            _ENGINE = HFEngine(model)
            load_ms = int((time.perf_counter() - loading) * 1000)

        sampling = time.perf_counter()
        images, _ = F.sample(hits[0], start, end, n=n_frames, max_side=640)
        sample_ms = int((time.perf_counter() - sampling) * 1000)
        if not images:
            return {"error": f"no frames decoded from {video_id} at [{start}, {end}]"}

        span = end - start
        generating = time.perf_counter()
        text = _ENGINE.generate(images, P.SYSTEM, P.user_prompt(span))
        ms = int((time.perf_counter() - generating) * 1000)

        events = P.parse_response(text, span) or []
        for event in events:                 # window-relative -> absolute video time
            if event.get("start") is not None:
                event["start"] = round(start + event["start"], 2)
                event["end"] = round(start + event["end"], 2)
        return {"events": events, "ms": ms, "sample_ms": sample_ms, "load_ms": load_ms,
                "cold_start": cold, "model": model, "frames": len(images),
                "window": [start, end], "video_id": video_id}
    except Exception as exc:
        # a reason the dashboard can show, rather than a 500 it cannot explain
        return {"error": f"{type(exc).__name__}: {exc}"}
