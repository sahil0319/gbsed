#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_gbsed_vs_image.py -- build the GBSED-vs-pixel comparison.

Reads the per-frame fidelity.csv files from both experiments and produces one
combined table plus the figures. Metrics are recomputed from the per-frame
rows rather than taken from the existing results_matrix.csv, because that
file's mean_edge_f1 column averages over ALL frames including LOST ones
(whose F1 is 0), which makes it algebraically identical to delivery_rate and
therefore carries no information of its own.

Usage:
    python tools/compare_gbsed_vs_image.py --out comparison
"""

import os
import csv
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RISKY = {"near_coll", "super_near"}
EGO = "ego car"

# Distinct channel conditions, ordered by severity. CAV_Good / CAV_Moderate /
# CAV_Bad are omitted: they set *.node[1].veinsmobility.x, which TraCI
# overrides, so they are byte-identical reruns of Baseline / Noise_Medium /
# Noise_High. The noise floor is the only thing that actually varied.
CONFIGS = [("Baseline", -98), ("Noise_Low", -95), ("Noise_Medium", -90),
           ("Noise_High", -85), ("CAV_Extreme", -80)]


def risky_count(edges):
    return sum(1 for s, r, d in edges if r in RISKY and (s == EGO or d == EGO))


def load_meta_risky(meta_dir):
    """frame -> number of safety-critical ego relations in the ground truth."""
    out = {}
    for f in glob.glob(str(Path(meta_dir) / "*.meta.json")):
        m = json.load(open(f))
        out[m["frame"]] = risky_count([tuple(e) for e in m["graph"]["edges"]])
    return out


def summarize(fid_csv, risky_by_frame=None, exact_means_perfect=None):
    """Recompute the metrics that matter from one run's per-frame rows.

    Both arms now write the same columns (gbsed_semantic.edge_metrics), so no
    arm-specific handling is needed.
    """
    rows = list(csv.DictReader(open(fid_csv)))
    n = len(rows)
    delivered = [r for r in rows if r["status"] != "LOST"]
    exact = [r for r in rows if r["status"] == "EXACT"]

    def col(r, name):
        v = r.get(name, "")
        return float(v) if v not in ("", None) else 0.0

    f1d = [col(r, "edge_f1") for r in delivered]
    af1d = [col(r, "actor_edge_f1") for r in delivered]
    r_orig = sum(int(col(r, "risky_orig")) for r in delivered)
    r_keep = sum(int(col(r, "risky_preserved")) for r in delivered)

    return {
        "frames": n,
        "delivered": len(delivered),
        "delivery_rate": len(delivered) / n if n else 0.0,
        "graph_exact": len(exact),
        "graph_exact_rate": len(exact) / n if n else 0.0,
        "mean_f1_delivered": sum(f1d) / len(f1d) if f1d else 0.0,
        "mean_f1_all": sum(col(r, "edge_f1") for r in rows) / n if n else 0.0,
        "mean_actor_f1_delivered": sum(af1d) / len(af1d) if af1d else 0.0,
        "risky_total": r_orig,
        "risky_preserved": r_keep,
        "risky_rate": r_keep / r_orig if r_orig else 0.0,
        "detections": sum(int(col(r, "n_detections")) for r in delivered),
        "psnr": np.mean([float(r["psnr_db"]) for r in delivered
                         if r.get("psnr_db")]) if delivered and
                         any(r.get("psnr_db") for r in delivered) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbsed-results", default="experiment_results")
    ap.add_argument("--image-results", default="experiment_results_image")
    ap.add_argument("--meta", default="scene_data_seq1")
    ap.add_argument("--seq", default="seq1")
    ap.add_argument("--out", default="comparison")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    risky_by_frame = load_meta_risky(args.meta)

    arms = [
        ("GBSED (scene graph)", lambda c:
            f"{args.gbsed_results}/decoded_{args.seq}_{c}/fidelity.csv", True),
        ("WebP @ same bytes", lambda c:
            f"{args.image_results}/decoded_image_data_{args.seq}_webp_{c}/fidelity.csv", False),
        ("JPEG @ same bytes", lambda c:
            f"{args.image_results}/decoded_image_data_{args.seq}_jpeg_{c}/fidelity.csv", False),
    ]

    table, missing = [], []
    for arm, pathf, perfect in arms:
        for cfg, nf in CONFIGS:
            p = pathf(cfg)
            if not os.path.isfile(p):
                missing.append(p)
                continue
            s = summarize(p, risky_by_frame, perfect)
            s.update({"arm": arm, "config": cfg, "noise_floor_dbm": nf})
            table.append(s)

    cols = ["arm", "config", "noise_floor_dbm", "frames", "delivered",
            "delivery_rate", "graph_exact", "graph_exact_rate",
            "mean_f1_delivered", "mean_f1_all", "mean_actor_f1_delivered",
            "risky_total", "risky_preserved", "risky_rate",
            "detections", "psnr"]
    with open(outdir / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in table:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ---------------------------------------------------------------- plots
    names = [c for c, _ in CONFIGS]
    xs = np.arange(len(names))
    arm_names = [a for a, _, _ in arms]
    colors = {"GBSED (scene graph)": "#1b7837",
              "WebP @ same bytes": "#d6604d",
              "JPEG @ same bytes": "#8c6bb1"}

    def series(arm, key):
        d = {r["config"]: r[key] for r in table if r["arm"] == arm}
        return [d.get(c, np.nan) for c in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    for a in arm_names:
        ax.plot(xs, [100 * v for v in series(a, "delivery_rate")], marker="o",
                color=colors[a], label=a, lw=2, ms=7,
                ls="--" if a != "GBSED (scene graph)" else "-")
    ax.set_title("Delivery rate\n(identical by construction)", fontsize=11)
    ax.set_ylabel("% of frames delivered"); ax.set_ylim(-4, 104)

    ax = axes[1]
    for a in arm_names:
        ax.plot(xs, series(a, "mean_actor_f1_delivered"), marker="o",
                color=colors[a], label=a, lw=2, ms=7,
                ls="--" if a != "GBSED (scene graph)" else "-")
    ax.set_title("Actor-relation F1 | delivered\n(skeleton excluded)", fontsize=11)
    ax.set_ylabel("mean F1"); ax.set_ylim(-0.04, 1.04)

    ax = axes[2]
    for a in arm_names:
        ax.plot(xs, [100 * v for v in series(a, "risky_rate")], marker="o",
                color=colors[a], label=a, lw=2, ms=7,
                ls="--" if a != "GBSED (scene graph)" else "-")
    ax.set_title("Safety-critical relations preserved\n(near_coll / super_near with ego)",
                 fontsize=11)
    ax.set_ylabel("% of relations"); ax.set_ylim(-4, 104)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(["%s\n%d dBm" % (n, nf) for n, (_, nf) in zip(names, CONFIGS)],
                           fontsize=8)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_xlabel("channel condition (noise floor)")
    fig.suptitle("Same bytes, same channel, same frames — only the representation differs",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "01_gbsed_vs_image.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- payload size, log scale
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels, vals = [], []
    for name, mdir in [("GBSED\nscene graph", args.meta),
                       ("WebP\n@ budget", "image_data_%s_webp" % args.seq),
                       ("JPEG\n@ budget", "image_data_%s_jpeg" % args.seq),
                       ("Full JPEG\n(1280x720)", "image_data_%s_full" % args.seq)]:
        mf = Path(mdir) / "manifest.json"
        if mf.is_file():
            m = json.load(open(mf))
            labels.append(name)
            vals.append(m.get("total_bytes") or m.get("gbsed_total_bytes"))
    bars = ax.bar(labels, vals, color=["#1b7837", "#d6604d", "#8c6bb1", "#4d4d4d"])
    ax.set_yscale("log"); ax.set_ylabel("total bytes for 20 frames (log)")
    ax.set_title("Payload size")
    base = vals[0]
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.25,
                f"{v:,}\n({v/base:.0f}x)" if v != base else f"{v:,}\n(1x)",
                ha="center", fontsize=9)
    ax.set_ylim(top=max(vals) * 6)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "02_payload_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("wrote %s" % (outdir / "comparison.csv"))
    print("wrote %s" % (outdir / "01_gbsed_vs_image.png"))
    print("wrote %s" % (outdir / "02_payload_size.png"))
    if missing:
        print("\nmissing (skipped):")
        for m in missing:
            print("  " + m)

    hdr = "%-22s %-13s %9s %11s %13s %9s %9s" % (
        "arm", "config", "delivered", "graph exact", "safety rels",
        "edge F1", "actor F1")
    print("\n" + hdr); print("-" * len(hdr))
    for r in table:
        print("%-22s %-13s %8.0f%% %10.0f%% %12s %9.3f %9.3f"
              % (r["arm"], r["config"], 100 * r["delivery_rate"],
                 100 * r["graph_exact_rate"],
                 "%d/%d" % (r["risky_preserved"], r["risky_total"]),
                 r["mean_f1_delivered"], r["mean_actor_f1_delivered"]))


if __name__ == "__main__":
    main()
