#!/usr/bin/env bash
# Bare cloud GPU box -> smoke-tested VLM inference. Paste this into the Lightning/Modal terminal.
# Safe to re-run; every step is idempotent.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"

echo "=== 1. GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || { echo "NO GPU VISIBLE"; exit 1; }

echo "=== 2. deps ==="
pip install -q --upgrade pip
pip install -q opencv-python-headless pillow requests
pip install -q "transformers>=4.57" accelerate peft qwen-vl-utils
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda OK')"

echo "=== 3. dataset ==="
if [ ! -d dataset/test/videos ]; then
  echo "dataset/ missing -- pulling from the Drive mirror (cloud->cloud, do NOT upload from a laptop)"
  pip install -q gdown
  gdown --folder https://drive.google.com/drive/folders/1sEFKR7ctd5GfFw-nMlYd_MnTw1VVYz9K -O . || \
    echo "gdown failed -- try another mirror from .context/01-dataset.md"
fi
ls dataset/test/videos/*.mp4 2>/dev/null | wc -l | xargs echo "test videos present:"

echo "=== 4. model ==="
pip install -q "huggingface_hub[cli]"
hf download "$MODEL" --quiet || huggingface-cli download "$MODEL"

echo "=== 5. does this backbone actually take video/multi-image? ==="
python - <<'PY'
import os
from transformers import AutoConfig, AutoProcessor
m = os.environ.get("MODEL", "Qwen/Qwen3-VL-4B-Instruct")
cfg = AutoConfig.from_pretrained(m, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(m, trust_remote_code=True)
print("model_type:", getattr(cfg, "model_type", "?"))
print("processor:", type(proc).__name__)
print("has video processor:", hasattr(proc, "video_processor") or "video" in str(type(proc)).lower())
print("image token:", getattr(proc, "image_token", None))
PY

echo "=== 6. smoke test: 2 videos, 8 frames ==="
PYTHONPATH=src python -m vad.run \
  --videos dataset/test/videos --manifest data/manifest.json \
  --engine hf --model "$MODEL" --frames 8 --limit 2 --out out/smoke

echo
echo "=== smoke output ==="
head -c 600 out/smoke.events.jsonl || true
echo
echo "DONE. Full zero-shot run:"
echo "  PYTHONPATH=src python -m vad.run --videos dataset/test/videos --manifest data/manifest.json \\"
echo "      --engine hf --model $MODEL --out out/zeroshot"
