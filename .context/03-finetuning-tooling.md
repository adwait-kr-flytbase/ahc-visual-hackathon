# Fine-tuning tooling — organiser recommendations

Three stacks named in the primer. **Docs read 2026-09-05 — see the verdict at the bottom.**

> ## ⚠️ Decisive fact: only ms-swift takes video natively
> Unsloth's vision fine-tuning docs cover **images only** — no video input path.
> ms-swift has a first-class `<video>` tag and video env-var controls.
> For a *video* anomaly task, **ms-swift is the default choice**; Unsloth is viable only if
> we reduce each clip to sampled frames ourselves and treat it as a multi-image problem.

---

## 1. Unsloth — fastest start
Free Colab notebooks exist for: **Qwen3-VL 8B, Gemma 3 4B, Qwen2.5-VL 7B, Llama 3.2 Vision 11B, Pixtral 12B.**

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers   = False,   # frozen encoder
    finetune_language_layers = True,
    r = 16, lora_alpha = 16,
    target_modules = "all-linear",
)
```

**Gotchas called out by the organisers (confirmed in the docs):**
- Use `UnslothVisionDataCollator` with `train_on_responses_only = True`.
- Build the dataset with a **list comprehension, NOT `dataset.map()`** — *"map kicks in dataset
  standardization and arrow processing rules which can be strict"*, which breaks multi-image samples.
  Use `[convert_to_conversation(s) for s in dataset]`.

**Dataset format** (images as a content-part list):
```python
[{"role": "user", "content": [{"type": "text", "text": instruction},
                              {"type": "image", "image": image}]},
 {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
```
Docs advise keeping images **300–1000px** to control resource use.
The docs do **not** take a position on freezing vision layers — the frozen-encoder setting in the
organisers' snippet is their recommendation, not Unsloth's.

**No video support documented.** Docs: https://unsloth.ai/docs/basics/vision-fine-tuning

---

## 2. ms-swift — CLI-driven, broader model coverage
Training + inference + eval + export in one tool.

```bash
swift sft --model Qwen/Qwen3-VL-4B-Instruct \
    --dataset train.jsonl --val_dataset val.jsonl \
    --tuner_type lora --lora_rank 8 --lora_alpha 32 \
    --freeze_vit true --freeze_aligner true \
    --torch_dtype bfloat16 --learning_rate 1e-4 \
    --num_train_epochs 1 --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 --gradient_checkpointing true \
    --max_length 4096 --output_dir output
```

**Video dataset format** — `<video>` tag in the message, path in a parallel `videos` array:
```json
{"messages": [{"role": "user", "content": "<video>what's in the video?"},
              {"role": "assistant", "content": "A puppy running on grass"}],
 "videos": ["/path/to/video.mp4"]}
```

**The memory/latency dials** (defaults from the Qwen3-VL best-practice guide):

| Env var | Purpose | Default |
|---|---|---|
| `FPS_MAX_FRAMES` | max frames extracted per video | **16** |
| `VIDEO_MAX_TOKEN_NUM` | token budget for the whole video | **128** |
| `VIDEO_MAX_PIXELS` | pixel budget per frame | **128×32×32** |
| `IMAGE_MAX_TOKEN_NUM` | image token limit | 1024 |

→ **16 frames / 128 tokens is a very tight budget.** A ~30s multi-event clip at 16 frames is ~0.5 fps
sampling — plausibly too coarse for a 1-second accident, and directly in tension with temporal
localisation (Levels 2–3). Treat these as first-class experiment axes, not defaults to accept.

**VRAM (from the guide):** Qwen3-VL-4B-Instruct LoRA ≈ **2×21 GB**. That does **not** fit a single
Kaggle T4 (16 GB) at these settings — expect to cut `FPS_MAX_FRAMES`/`VIDEO_MAX_PIXELS`, or use
Kaggle's T4**×2**, or Lightning's L4/L40S. Qwen3-VL-30B-A3B full-parameter is 8×80 GB — out of scope.

**Inference:**
```bash
swift infer --model Qwen/Qwen3-VL-4B-Instruct --stream true
swift infer --adapters output/checkpoint-xxx --stream true --max_new_tokens 2048
```

Docs: https://swift.readthedocs.io/en/latest/ ·
[dataset formats](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html) ·
[Qwen3-VL best practice](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html)

---

## 3. HF TRL + PEFT — the reference stack
**Gotcha:** set `max_length=None` in `SFTConfig`, or truncation **silently cuts image tokens.**

Docs: [TRL VLM SFT](https://huggingface.co/docs/trl/main/en/training_vlm_sft) ·
[HF Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)

---

## Signal to read from this list

- Every example **freezes the vision encoder** and tunes only language/adapter layers
  (`finetune_vision_layers=False`, `--freeze_vit true --freeze_aligner true`).
  That's the organisers' expected default: LoRA on the LM side, encoder untouched.
- Model sizes implied as tractable: **4B–12B**, with **Qwen3-VL-4B** appearing as the ms-swift default
  and Gemma 3 4B in the Unsloth list. That's the practical "small VLM" band for this event.
- All three gotchas are about **image/video tokens being silently dropped or mishandled** —
  the recurring failure mode in VLM fine-tuning. Verify token counts before trusting a training run.

## Verdict
**ms-swift + Qwen3-VL-4B, LoRA, frozen ViT + frozen aligner** is the path of least resistance: native
video, the organisers' own example command, and the smallest model in the implied band. Falsify the
16-frame/128-token budget early — it is the binding constraint on temporal localisation, and it is the
first thing to tune, not the last.
