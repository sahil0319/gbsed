#!/usr/bin/env python3
"""Plot the payload-layout sweep: does slice-aligned packing preserve the
relations that matter when a frame only partly arrives?"""
import csv, argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="experiment_results_format/format_results.csv")
ap.add_argument("--out", default="experiment_results_format")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.csv)))
CONFIGS = ["Baseline", "Noise_Low", "Noise_Medium", "Noise_High", "CAV_Extreme"]
NF = [-98, -95, -90, -85, -80]

def series(arm, key):
    d = {r["config"]: r for r in rows if r["arm"] == arm}
    out = []
    for c in CONFIGS:
        v = d.get(c, {}).get(key, "")
        out.append(float(v) if v not in ("", None) else np.nan)
    return out

xs = np.arange(len(CONFIGS))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

style = {
    "v1_cs1000": ("#9ecae1", "flat serialisation, 1000 B chunks", "--", "s"),
    "v2_cs1000": ("#31a354", "slice-aligned, 1000 B chunks", "--", "^"),
    "v1_cs500":  ("#d6604d", "flat serialisation, 500 B chunks", "-", "s"),
    "v2_cs500":  ("#1b7837", "slice-aligned, 500 B chunks", "-", "o"),
}

for ax, key, title, ylab in [
    (axes[0], "risky_recall", "Safety-critical relations preserved",
     "fraction of near_coll / super_near with ego"),
    (axes[1], "mean_actor_f1_delivered", "Actor-relation F1 | delivered",
     "mean F1, skeleton excluded"),
]:
    for arm, (col, lab, ls, mk) in style.items():
        ax.plot(xs, series(arm, key), color=col, label=lab, ls=ls,
                marker=mk, lw=2, ms=7, alpha=0.95)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%s\n%d dBm" % (c, n) for c, n in zip(CONFIGS, NF)], fontsize=8)
    ax.set_ylim(-0.04, 1.08); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=11); ax.set_ylabel(ylab, fontsize=9)
    ax.set_xlabel("channel condition (noise floor)")
    ax.legend(fontsize=8, loc="lower left")

fig.suptitle("Slice-aligned packing: identical bytes and identical delivery, "
             "but a partial frame keeps what matters", fontsize=12.5)
fig.tight_layout()
out = Path(a.out) / "01_format_sweep.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

# At 1000 B nearly every frame is one chunk, so the two formats coincide;
# the separation only appears once frames span blocks.
print("\nwhere the formats differ (same delivery, different content):")
for c in CONFIGS:
    a1 = next((r for r in rows if r["arm"] == "v1_cs500" and r["config"] == c), None)
    a2 = next((r for r in rows if r["arm"] == "v2_cs500" and r["config"] == c), None)
    if a1 and a2 and a1["risky_total"] != "0":
        print("  %-13s delivered %s both;  safety flat %s/%s -> slice-aligned %s/%s"
              % (c, a1["delivered"], a1["risky_preserved"], a1["risky_total"],
                 a2["risky_preserved"], a2["risky_total"]))
