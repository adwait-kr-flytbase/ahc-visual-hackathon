"""Two backends behind one interface. Start on HF today; swap to the vLLM server later
without touching anything upstream."""
import base64, io, os


class HFEngine:
    """transformers. Works immediately on any GPU. Serial."""

    def __init__(self, model_id, adapter=None, device="cuda", max_new_tokens=384):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as Model
        except ImportError:
            from transformers import AutoModelForVision2Seq as Model
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Model.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.name = os.path.basename(adapter.rstrip("/")) if adapter else model_id
        self.concurrent = False

    def generate(self, frames, system, user):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image"} for _ in frames] + [{"type": "text", "text": user}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=frames, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)


class ServerEngine:
    """Any OpenAI-compatible endpoint: `vllm serve <model>`, or NIM/Gemini for dev-time comparison.
    Thread-safe, so run.py can fire windows concurrently."""

    def __init__(self, model, base_url="http://localhost:8000/v1", api_key=None, max_new_tokens=384):
        import requests
        self.requests = requests
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.max_new_tokens = max_new_tokens
        self.name = model
        self.concurrent = True

    @staticmethod
    def _b64(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def generate(self, frames, system, user):
        content = [{"type": "image_url", "image_url": {"url": self._b64(f)}} for f in frames]
        content.append({"type": "text", "text": user})
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
        }
        r = self.requests.post(self.url, json=body, timeout=300,
                               headers={"Authorization": f"Bearer {self.key}"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class GeminiEngine:
    """Google's native generateContent API. Frames go inline as JPEG parts.

    Dev-time REFERENCE CEILING only -- a hosted model can never be in the runtime path.
    """

    def __init__(self, model="gemini-3.5-flash", api_key=None, max_new_tokens=2048):
        import requests
        self.requests = requests
        self.model = model
        self.key = api_key or os.environ["GEMINI_API_KEY"]
        self.url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent")
        self.max_new_tokens = max_new_tokens
        self.name = model
        self.concurrent = True

    @staticmethod
    def _b64(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def generate(self, frames, system, user):
        parts = [{"inline_data": {"mime_type": "image/jpeg", "data": self._b64(f)}}
                 for f in frames]
        parts.append({"text": user})
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": self.max_new_tokens,
                "responseMimeType": "application/json",
                # Gemini 3.x is a reasoning model: thinking tokens are drawn from
                # maxOutputTokens. With a small budget it thinks, then returns a
                # truncated fragment or a candidate with no parts at all.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        import random, time
        last = None
        for attempt in range(5):                      # 429/503 are routine on the free tier
            r = self.requests.post(self.url, json=body, timeout=300,
                                   headers={"x-goog-api-key": self.key})
            if r.status_code < 400:
                break
            last = r
            if r.status_code in (429, 500, 503, 504):
                time.sleep(min(2 ** attempt + random.random(), 30))
                continue
            r.raise_for_status()
        else:
            last.raise_for_status()
        d = r.json()
        cands = d.get("candidates") or []
        if not cands:
            return ""
        c = cands[0]
        parts = (c.get("content") or {}).get("parts") or []
        if not parts:                      # safety block, MAX_TOKENS, or empty candidate
            return f"__NO_PARTS__ finishReason={c.get('finishReason')}"
        return "".join(p.get("text", "") for p in parts)


def build(kind, **kw):
    return HFEngine(**kw) if kind == "hf" else ServerEngine(**kw)
