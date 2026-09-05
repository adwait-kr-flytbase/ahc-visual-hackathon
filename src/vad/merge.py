"""Window predictions -> video-level events.

This is where D2/D3 marks are won or lost, and it costs zero GPU.
Every constant here is swept by sweep.py.
"""
from .prompts import CLASSES

DEFAULTS = {
    "min_conf": 0.35,      # global confidence floor
    "gap_tol": 6.0,        # merge same-class events separated by <= this many seconds
    "min_dur": 1.0,        # drop events shorter than this
    "pad": 0.0,            # symmetric boundary padding
    "max_events": 12,      # per video
}


def merge(window_events, duration, cfg=None, per_class_conf=None):
    """window_events: dicts with ABSOLUTE start/end, class_name, confidence, explanation."""
    c = dict(DEFAULTS)
    if cfg:
        c.update(cfg)
    pcc = per_class_conf or {}

    kept = [e for e in window_events
            if e["confidence"] >= max(c["min_conf"], pcc.get(e["class_name"], 0.0))]

    out = []
    for cls in CLASSES:
        xs = sorted([e for e in kept if e["class_name"] == cls], key=lambda e: e["start"])
        cur = None
        for e in xs:
            if cur is None:
                cur = dict(e)
                continue
            if e["start"] <= cur["end"] + c["gap_tol"]:
                cur["end"] = max(cur["end"], e["end"])
                if e["confidence"] > cur["confidence"]:
                    cur["confidence"] = e["confidence"]
                    cur["explanation"] = e["explanation"]
            else:
                out.append(cur)
                cur = dict(e)
        if cur is not None:
            out.append(cur)

    final = []
    for e in out:
        s = max(0.0, e["start"] - c["pad"])
        t = min(duration, e["end"] + c["pad"])
        if t - s < c["min_dur"]:
            continue
        final.append({**e, "start": round(s, 2), "end": round(t, 2)})

    final.sort(key=lambda e: -e["confidence"])
    final = final[: int(c["max_events"])]
    final.sort(key=lambda e: e["start"])
    return final
