# Compute, model access, and setup

## Coding setup
Any coding agent (Claude Code, OpenCode, Codex, Cursor…) with your own model access is fine.
The doc's walkthrough — Claude Code + OpenRouter + a free model — is *an example, not a requirement*.

If routing Claude Code through OpenRouter:
```bash
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="sk-or-v1-..."
export ANTHROPIC_MODEL="nvidia/nemotron-3.5-lightning:free"
claude -p "Reply with exactly: OK"     # verify
```
Free models: https://openrouter.ai/models?max_price=0&output_modalities=text
Env vars are per-terminal-session. Have 1–2 backup model IDs ready — free models get rate-limited.

> **Not relevant to this session** — Claude Code is already running on a paid Opus 5 setup.
> Recorded only for completeness.

---

## Part 1 — Free GPU runtimes (for running/training your own models)

| Option | GPU | Allowance | Notes |
|---|---|---|---|
| **Kaggle Notebooks** | T4 ×2 | **30 GPU-hrs/week** | Requires **phone verification** or the GPU option stays locked. Save to `/kaggle/working/`. |
| **Google Colab** | T4 | variable/free tier | Runtime → Change runtime type → T4 GPU. Mount Drive to persist. |
| **Lightning AI** | T4 / L4 / L40S / A100 | 5 credits free, **+25 with a card = ~$30** | Persistent storage. Prep on free CPU instance, switch to GPU only to train. |
| **Modal** | serverless, per-second billing | **$30/month** compute credit | **Card required on file** to use at all. Credits reset monthly, don't roll over. Set a Budget. |

Always verify: `import torch; print(torch.cuda.is_available())` → must be `True`.

**Kaggle's 30 GPU-hrs/week is the best free allowance** and T4×2 is a reasonable fine-tuning target
for a 4B VLM with LoRA + frozen ViT. Lightning's L40S/A100 is the escape hatch if T4 VRAM binds.

Modal setup:
```bash
pipx install modal   # or pip install modal
modal setup
modal run gpu_test.py
```

---

## Part 2 — Hosted model APIs (dev-time only — cannot be in the runtime path)

| Option | Access | Limits |
|---|---|---|
| **AI Grants India × FlytBase** | Form link given **on hackathon day** → WhatsApp → key for OpenAI `gpt-5.6-luna` | **~4 RPM** — heavily rate limited |
| **NVIDIA NIM** | build.nvidia.com, free, no card, phone verification required | ~40 RPM. OpenAI-SDK compatible via `base_url="https://integrate.api.nvidia.com/v1"`. [Vision catalog](https://build.nvidia.com/explore/vision) |
| **Gemini API** | aistudio.google.com, free, no card | **Flash / Flash-Lite only** (Pro needs billing). Limits are **per project, not per key**. `429` → back off. |

**Gemini accepts video and images directly** — that makes it the natural choice for
bulk pseudo-labelling of unannotated drone footage. NVIDIA NIM's 40 RPM is 10× the AI Grants key,
so NIM + Gemini are the two workhorses for label generation.

---

## Pre-event checklist (from the doc)
- [ ] Kaggle account created **and phone-verified**
- [ ] Colab opened at least once
- [ ] Modal account set up (card on file)
- [ ] NVIDIA API key generated and saved
- [ ] Gemini API key generated and saved
- [ ] One successful test call made with each
