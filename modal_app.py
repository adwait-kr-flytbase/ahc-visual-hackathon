"""Modal driver for the AHC VAD build.

  .venv/bin/modal run modal_app.py::pull_dataset          # 15GB Drive -> Volume, cloud-to-cloud
  .venv/bin/modal run modal_app.py::zeroshot              # zero-shot on the 34 public videos
  .venv/bin/modal run modal_app.py::sft --cmd "$(cat cmd)"  # run a swift command verbatim
"""
import modal

app = modal.App("ahc-vad")
vol = modal.Volume.from_name("ahc-vad-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("ahc-vad-hf", create_if_missing=True)

BASE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git")
    .pip_install("opencv-python-headless", "pillow", "requests", "numpy")
)

CORE = BASE.pip_install(
    "torch==2.6.0", "torchvision",
    "transformers>=4.57", "accelerate", "peft", "qwen-vl-utils",
    "huggingface_hub[hf_transfer]",
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf", "PYTHONPATH": "/root/src"})

TRAIN_CORE = CORE.pip_install("ms-swift", "trl", "datasets", "av", "decord")

# add_local_* must come LAST in every image chain -- Modal forbids build steps after it.
LOCAL = lambda img: img.add_local_dir("src", "/root/src").add_local_dir("data", "/root/data")

INFER = LOCAL(CORE)
TRAIN = LOCAL(TRAIN_CORE)

PULL = BASE.pip_install("gdown").env({"PYTHONPATH": "/root/src"})

MIRRORS = [
    "1sEFKR7ctd5GfFw-nMlYd_MnTw1VVYz9K",
    "13E_CePn14lcbwMA_yZEiHpAVx6i09UIG",
    "13V8JqgZRMzn2TCF0HTsCqVgUH0UOMmpb",
]


@app.function(image=PULL, volumes={"/vol": vol}, timeout=7200, cpu=4.0)
def pull_dataset():
    """Pull the 15GB pack straight into the Volume. Never upload it from a laptop."""
    import subprocess, os
    os.makedirs("/vol/dataset", exist_ok=True)
    for mid in MIRRORS:
        print(f"--- trying mirror {mid}", flush=True)
        r = subprocess.run(
            ["gdown", "--folder", "-O", "/vol/dataset", "--remaining-ok",
             f"https://drive.google.com/drive/folders/{mid}"],
            capture_output=True, text=True, timeout=6000)
        print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
        n = subprocess.run("find /vol/dataset -name '*.mp4' | wc -l", shell=True,
                           capture_output=True, text=True).stdout.strip()
        print(f"mirror {mid}: {n} mp4 files now on the volume", flush=True)
        if int(n or 0) > 3000:
            break
    vol.commit()
    return subprocess.run("du -sh /vol/dataset; find /vol/dataset -name '*.mp4' | wc -l",
                          shell=True, capture_output=True, text=True).stdout


def _stream(cmd, cwd, shell):
    """Run a child process with LIVE output and periodic volume commits.

    capture_output=True buffers everything until exit, which leaves a 40-minute GPU job
    completely unobservable. Learned that the hard way.
    """
    import subprocess, time
    proc = subprocess.Popen(cmd, cwd=cwd, shell=shell, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines, last = [], time.time()
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        lines.append(line)
        if time.time() - last > 60:      # flush partial results to the volume every minute
            try:
                vol.commit()
            except Exception as e:
                print(f"(commit skipped: {e})", flush=True)
            last = time.time()
    proc.wait()
    vol.commit()
    return {"rc": proc.returncode, "tail": "\n".join(lines[-40:])}


@app.function(image=INFER, gpu="A100-40GB", volumes={"/vol": vol, "/hf": hf_cache},
              timeout=7200, secrets=[modal.Secret.from_name("hf-token")])
def zeroshot(model: str = "Qwen/Qwen3-VL-4B-Instruct",
             videos: str = "/vol/dataset/test/videos",
             manifest: str = "/root/data/manifest.json",
             out: str = "/vol/out/zeroshot",
             frames: int = 16, win: float = 20.0, hop: float = 10.0,
             limit: int = 0, skip: int = 0, adapter: str = "", variant: str = "default"):
    import subprocess, sys, os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv",
                         shell=True, capture_output=True, text=True).stdout, flush=True)
    cmd = [sys.executable, "-m", "vad.run", "--videos", videos, "--manifest", manifest,
           "--engine", "hf", "--model", model, "--out", out,
           "--frames", str(frames), "--win", str(win), "--hop", str(hop)]
    if limit:
        cmd += ["--limit", str(limit)]
    if skip:
        cmd += ["--skip", str(skip)]
    if variant != "default":
        cmd += ["--variant", variant]
    if adapter:
        cmd += ["--adapter", adapter]
    print(" ".join(cmd), flush=True)
    return _stream(cmd, cwd="/root", shell=False)


@app.function(image=TRAIN, gpu="A100-80GB", volumes={"/vol": vol, "/hf": hf_cache},
              timeout=14400, secrets=[modal.Secret.from_name("hf-token")])
def sft(cmd: str):
    """Run a swift command verbatim. No hyperparameter second-guessing."""
    import subprocess
    print(subprocess.run("nvidia-smi; which swift || pip show ms-swift | head -3",
                         shell=True, capture_output=True, text=True).stdout, flush=True)
    import os, subprocess as sp
    if os.path.exists("/vol/sft/windows.tar") and not os.path.isdir("/vol/sft/windows"):
        print("untarring window clips...", flush=True)
        sp.run("tar -xf /vol/sft/windows.tar -C /vol/sft", shell=True, check=True)
        n = sp.run("ls /vol/sft/windows | wc -l", shell=True, capture_output=True, text=True).stdout.strip()
        print(f"untarred: {n} window dirs", flush=True)
        vol.commit()
    print("RUNNING:\n" + cmd, flush=True)
    return _stream(cmd, cwd="/vol", shell=True)


@app.function(image=BASE, volumes={"/vol": vol}, timeout=1800)
def ls(path: str = "/vol"):
    import subprocess
    return subprocess.run(f"du -sh {path}/* 2>/dev/null; echo ---; find {path} -name '*.mp4' | wc -l",
                          shell=True, capture_output=True, text=True).stdout


@app.local_entrypoint()
def main(action: str = "zeroshot", limit: int = 0, frames: int = 16,
         model: str = "Qwen/Qwen3-VL-4B-Instruct", out: str = "/vol/out/zeroshot",
         skip: int = 0, variant: str = "default", win: float = 20.0, hop: float = 10.0):
    if action == "pull":
        print(pull_dataset.remote())
    elif action == "ls":
        print(ls.remote())
    else:
        print(zeroshot.remote(model=model, frames=frames, limit=limit, out=out, skip=skip, variant=variant, win=win, hop=hop))
