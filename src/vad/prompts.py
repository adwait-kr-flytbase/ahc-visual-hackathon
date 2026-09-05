"""SHARED CONTRACT — imported by BOTH the training lane and the inference lane.

Do not fork this file. If the training prompt and the inference prompt drift,
the fine-tuned model will score worse than zero-shot.
"""
import json, re

from ahc_vad.taxonomy import ANOMALY_CLASSES  # single source of truth (src/ahc_vad/taxonomy.py)

# Ordered view of ANOMALY_CLASSES. The set is unordered; prompts and merge iteration need a
# stable order, so the list lives here and is checked against the canonical set below.
CLASSES = [
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "wrong_way_driving",
    "road_spill_or_debris",
    "fire",
    "smoke",
    "waterlogging_or_flood",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
]
CLASS_SET = set(CLASSES)
assert CLASS_SET == set(ANOMALY_CLASSES), "vad.prompts.CLASSES has drifted from ahc_vad.taxonomy"

SYSTEM = (
    "You are a traffic and public-safety video anomaly detector. "
    "You watch a video segment from a drone, dashcam or CCTV camera and report anomalous events. "
    "You answer with JSON only, no prose."
)

# ASK-HINT style: grouped, with the discriminative cue for each class.
GUIDE = """Anomaly classes, grouped:

TRAFFIC
- traffic_accident: a collision or crash, impact, overturned or wrecked vehicle
- traffic_congestion: a dense queue of slow or stopped vehicles
- stalled_or_broken_down_vehicle: a vehicle stationary on the carriageway or shoulder where it should not be
- vehicle_blocking_traffic: a vehicle obstructing a lane so others must stop or divert
- wrong_way_driving: a vehicle moving against the dominant direction of flow

HAZARD
- fire: visible flames
- smoke: a smoke plume with no clear flames
- waterlogging_or_flood: standing water covering the road or ground
- road_spill_or_debris: spilled load, debris or objects lying on the carriageway

BEHAVIOUR
- fighting_or_violence: physical fighting, assault, punching or kicking
- loitering_or_suspicious_presence: a person lingering, prowling or trespassing with no clear purpose"""


# Experiment: the D1 baseline is precision-heavy (P=0.77, R=0.50) and returns NOTHING on
# 8 of 24 videos, all of them context/duration classes. These variants probe whether the
# silence is a prompting artifact or a genuine capability limit.
VARIANTS = {
    "default": "",
    "recall": (
        "\n\nIMPORTANT: subtle events count. A vehicle that has been stationary where it should "
        "not be, an object lying on the carriageway, people in a physical altercation, a person "
        "lingering with no purpose, a vehicle obstructing others — these are anomalies even when "
        "the scene looks otherwise ordinary. Do not require the event to be dramatic. If you see a "
        "plausible anomaly, report it with a confidence that reflects your certainty rather than "
        "staying silent."
    ),
    "behaviour": (
        "\n\nFOCUS ONLY on people. Is anyone LINGERING with no clear purpose, loitering, "
        "prowling or trespassing? Is anyone FIGHTING — pushing, punching, kicking, grappling? "
        "Report only loitering_or_suspicious_presence or fighting_or_violence, or an empty list."
    ),
    "roadstate": (
        "\n\nFOCUS ONLY on the road surface and stationary objects. Is there DEBRIS, a spilled "
        "load, or an object lying on the carriageway? Is a vehicle STOPPED where it should not be — "
        "on a shoulder, in a live lane, blocking others? Report only road_spill_or_debris, "
        "stalled_or_broken_down_vehicle or vehicle_blocking_traffic, or an empty list."
    ),
    "forced": (
        "\n\nYou must commit to a judgement. First decide: is anything in this segment "
        "out of the ordinary for this scene? If yes, name the single best-matching class even if "
        "you are unsure, and set confidence accordingly. Only return an empty list if the scene is "
        "genuinely routine."
    ),
}


def user_prompt(window_sec: float, variant: str = "default") -> str:
    return (
        f"{GUIDE}\n\n"
        f"This segment is {window_sec:.1f} seconds long.\n"
        f"Report every anomalous event visible in it. Times are in seconds RELATIVE to the start "
        f"of THIS segment and must lie between 0 and {window_sec:.1f}.\n"
        f"If nothing anomalous is happening, return an empty list.\n\n"
        'Answer with JSON only, exactly this shape:\n'
        '{"events": [{"class_name": "<one of the classes above>", "start": <float>, '
        '"end": <float>, "confidence": <0.0-1.0>, "explanation": "<one short sentence>"}]}'
        + VARIANTS.get(variant, "")
    )


def build_response(events) -> str:
    """Canonical assistant target. The training lane MUST use this to write SFT labels."""
    out = []
    for e in events:
        out.append({
            "class_name": e["class_name"],
            "start": round(float(e["start"]), 2),
            "end": round(float(e["end"]), 2),
            "confidence": round(float(e.get("confidence", 1.0)), 2),
            "explanation": (e.get("explanation") or "")[:300],
        })
    return json.dumps({"events": out}, ensure_ascii=False)


def build_sft_sample(video_path: str, events, window_sec: float) -> dict:
    """One ms-swift jsonl row. The training lane should call this, not hand-roll the template."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "<video>" + user_prompt(window_sec)},
            {"role": "assistant", "content": build_response(events)},
        ],
        "videos": [video_path],
    }


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_response(text: str, window_sec: float):
    """Model output -> list of events. Never raises; a bad generation yields []."""
    if not text:
        return []
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    m = _JSON_RE.search(text)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    raw = obj.get("events") if isinstance(obj, dict) else obj
    if not isinstance(raw, list):
        return []
    out = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        c = str(e.get("class_name", "")).strip()
        if c not in CLASS_SET:
            continue
        try:
            s = float(e.get("start", 0.0) or 0.0)
            t = float(e.get("end", window_sec) or window_sec)
        except Exception:
            s, t = 0.0, window_sec
        s = max(0.0, min(s, window_sec))
        t = max(0.0, min(t, window_sec))
        if t <= s:
            t = min(window_sec, s + 1.0)
        try:
            conf = float(e.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        out.append({
            "class_name": c,
            "start": s,
            "end": t,
            "confidence": max(0.0, min(1.0, conf)),
            "explanation": str(e.get("explanation") or "")[:480],
        })
    return out
