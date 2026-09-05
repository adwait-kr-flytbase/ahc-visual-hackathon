"""Two figures for the deck: the result that worked, and the one that did not."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ahc_vad.groundtruth import load_ground_truth, load_manifest  # noqa: E402

INK, PANEL, TEXT, DIM = "#0B0E13", "#141922", "#E4E9F1", "#7E8899"
HIT, MISS, FA = "#35D6B0", "#F2A93B", "#FF4D7D"
plt.rcParams.update({
    "figure.facecolor": INK, "axes.facecolor": PANEL, "savefig.facecolor": INK,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": DIM, "ytick.color": DIM,
    "axes.edgecolor": "#262E3B", "font.size": 9, "axes.titlesize": 10,
    "font.family": "monospace",
})

manifest = load_manifest(ROOT / "data/manifest.json")
gt = load_ground_truth(ROOT / "dataset/test/ground_truth.csv")
videos = {}
for line in (ROOT / "mv/out/features.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        if not r.get("failed"):
            videos[r["video_id"]] = r

def robust_z(v):
    v = np.asarray(v, float); m = np.median(v)
    mad = np.median(np.abs(v - m))
    return (v - m) / (mad * 1.4826 if mad > 1e-9 else (v.std() or 1.0))

# ---------- figure 1: ego-motion, per second rather than per video ----------
# The median over a whole video is the wrong statistic for D2/D3: they are composed from several
# source clips, so a dashcam stretch inside a mostly-static video disappears into the median.
# T033 is the case that caught this -- median 1.4e-4, yet its opening 40s reads 2.8e-3.
MOVING = 5e-4
frac = []
for v in videos:
    series = videos[v]["series"]
    if not series:
        continue
    ego = np.array([s["ego"] for s in series], float)
    frac.append((v, float((ego > MOVING).mean()), float(np.median(ego)), float(np.percentile(ego, 90))))
frac.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(11, 3.6))
names = [f[0] for f in frac]
vals = [f[1] * 100 for f in frac]
colours = [DIM if f[1] < .05 else (MISS if f[1] < .5 else HIT) for f in frac]
ax.bar(range(len(vals)), vals, color=colours, width=.72)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
ax.set_ylabel("% of seconds with a\nmoving camera")
ax.set_ylim(0, 105)
ax.set_title("Ego-motion, measured per second, comes free from the decoder. It is a continuum, not "
             "two classes:\n"
             f"{sum(1 for f in frac if f[1] < .05)} videos barely move, "
             f"{sum(1 for f in frac if .05 <= f[1] < .5)} move part of the time, "
             f"{sum(1 for f in frac if f[1] >= .5)} move throughout. "
             "The manifest's `domain` field is empty for all 34.",
             loc="left", color=TEXT, fontsize=9)
for spine in ("top", "right"): ax.spines[spine].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "mv/out/ego-motion.png", dpi=170); plt.close(fig)
print("per-second moving fraction, most to least:")
for v, f, med, p90 in sorted(frac, key=lambda x: -x[1])[:8]:
    print(f"  {v}  {f*100:5.1f}% of seconds moving   median {med:.5f}  p90 {p90:.5f}")

# ---------- figure 2: residual motion does not mark the events ----------
picks = ["T025", "T031", "T033"]
fig, axes = plt.subplots(len(picks), 1, figsize=(10, 5.6), sharex=False)
for ax, vid in zip(axes, picks):
    series = videos[vid]["series"]
    t = np.array([s["t"] for s in series], float)
    z = robust_z([s["residual"] for s in series])
    ax.plot(t, z, color=TEXT, lw=.9)
    ax.axhline(3, color=FA, lw=.9, ls="--")
    for e in gt.get(vid, []):
        if e.is_localised:
            ax.axvspan(e.start_time_sec, e.end_time_sec, color=HIT, alpha=.20, lw=0)
    ax.set_ylabel(f"{vid}\nresidual z", fontsize=8)
    ax.set_xlim(t.min(), t.max())
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)
axes[0].set_title("Residual motion energy vs ground-truth events (shaded). "
                  "Mean AUC 0.556 over 8 timed videos — chance is 0.500.",
                  loc="left", color=TEXT)
axes[-1].set_xlabel("seconds")
fig.tight_layout(); fig.savefig(ROOT / "mv/out/residual-vs-events.png", dpi=170); plt.close(fig)
print("wrote mv/out/ego-motion.png and mv/out/residual-vs-events.png")

# ---------- does it work better when the camera moves? ----------
report = json.loads((ROOT / "mv/out/report.json").read_text())
ego_by = {e["video_id"]: e["ego"] for e in report["ego"]}
print("\nAUC split by camera motion (the whole premise of the experiment):")
for label, test in (("static camera", lambda e: e <= 5e-4), ("moving camera", lambda e: e > 5e-4)):
    vals = [r["auc"]["residual"] for r in report["per_video"]
            if r["auc"]["residual"] is not None and test(ego_by[r["video_id"]])]
    print(f"  {label:15} n={len(vals)}  mean AUC {np.mean(vals):.3f}" if vals else f"  {label}: none")
