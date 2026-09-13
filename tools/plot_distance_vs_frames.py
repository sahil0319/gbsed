#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_distance_vs_frames.py -- vehicle separation vs. frame delivery.

For a given (sequence, channel-config) run, plots each frame's transmission
against the distance between the two vehicles at that moment, distinguishing
delivered frames from lost ones. This is the figure that makes the "range
cliff" concrete: fidelity does not decay smoothly with distance in a VANET --
frames arrive intact right up to a threshold, then stop arriving at all.

Data source
-----------
Reads <results_dir>/decoded_<seq>_<config>/fidelity.csv, produced by
gbsed_decode.py during the channel sweep (see run_experiment_matrix.sh).  That
file already carries, per frame:

    frame        -- frame index (x-axis)
    status       -- EXACT / DEGRADED / CORRUPT / LOST
    tx_time      -- simulation time the chunk was sent (s)
    distance_m   -- vehicle separation at RECEPTION, only populated for
                    frames that were actually heard

Lost frames have no measured distance (nothing was received to measure it
from). Their distance is estimated by fitting separation = m * tx_time + b
over the delivered frames -- which is highly linear in this scenario, since
both vehicles move at constant speed -- and evaluating that fit at the lost
frame's own tx_time. This is the same technique used for report/figures/
scenario.png, and it is reported as an estimate, not measured data.

Usage
-----
    python tools/plot_distance_vs_frames.py
    python tools/plot_distance_vs_frames.py --seq seq2 --config Noise_Low
    python tools/plot_distance_vs_frames.py --results-dir experiment_results_format \\
        --seq v2_cs500 --config Noise_Low
"""

import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN = "#1b7837"
RED = "#d6604d"
ORANGE = "#d95f02"


def load_fidelity(results_dir, seq, config):
    path = Path(results_dir) / f"decoded_{seq}_{config}" / "fidelity.csv"
    if not path.is_file():
        raise SystemExit(
            f"no such file: {path}\n"
            f"(run the channel sweep first -- see run_experiment_matrix.sh or "
            f"run_format_sweep.sh -- or pass --results-dir/--seq/--config to "
            f"point at an existing run)")
    # Some fidelity.csv files carry trailing blank rows; drop anything without
    # a frame index rather than letting int() fail on them.
    rows = [r for r in csv.DictReader(open(path)) if r.get("frame", "").strip()]
    if not rows:
        raise SystemExit(f"{path} has no usable rows")
    return rows, path


def main():
    ap = argparse.ArgumentParser(
        description="Plot vehicle separation vs. frame delivery for one run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--results-dir", default="experiment_results",
                    help="directory holding decoded_<seq>_<config>/fidelity.csv")
    ap.add_argument("--seq", default="seq1", help="sequence name, e.g. seq1")
    ap.add_argument("--config", default="Baseline",
                    help="channel config name, e.g. Baseline, Noise_Low")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: report/figures/distance_vs_frames_"
                         "<seq>_<config>.png)")
    args = ap.parse_args()

    rows, src = load_fidelity(args.results_dir, args.seq, args.config)

    frames = [int(r["frame"]) for r in rows]
    delivered = [r for r in rows if r["status"] != "LOST"]
    lost = [r for r in rows if r["status"] == "LOST"]

    if not delivered:
        raise SystemExit(f"no delivered frames in {src} -- nothing to fit or plot")

    # Fit separation = m * tx_time + b from delivered frames (measured
    # distance), then evaluate it at lost frames' own tx_time to estimate
    # what their separation would have been.
    dt = np.array([float(r["tx_time"]) for r in delivered])
    dd = np.array([float(r["distance_m"]) for r in delivered])
    if len(dt) >= 2:
        m, b = np.polyfit(dt, dd, 1)
        r = np.corrcoef(dt, dd)[0, 1]
    else:
        m, b, r = 0.0, float(dd[0]), float("nan")

    lost_frames, lost_dist = [], []
    for row in lost:
        if row.get("tx_time"):
            lost_frames.append(int(row["frame"]))
            lost_dist.append(m * float(row["tx_time"]) + b)

    deliv_frames = [int(r["frame"]) for r in delivered]
    deliv_dist = [float(r["distance_m"]) for r in delivered]
    cliff = max(deliv_dist) if deliv_dist else None

    # ---------------------------------------------------------------- plot
    fig, ax = plt.subplots(figsize=(10, 5.8))

    ax.plot(deliv_frames, deliv_dist, "o-", color=GREEN, lw=2.2, ms=9,
            zorder=4, label=f"delivered ({len(deliv_frames)})")
    if lost_frames:
        ax.scatter(lost_frames, lost_dist, s=90, facecolors="none",
                   edgecolors=RED, linewidths=2, zorder=4,
                   label=f"lost, estimated separation ({len(lost_frames)})")

    if cliff is not None:
        ax.axhline(cliff, color=ORANGE, ls="--", lw=1.6, zorder=2)
        ax.annotate(f"last delivered: {cliff:.0f} m", (min(frames), cliff),
                    textcoords="offset points", xytext=(4, 6), fontsize=10,
                    color=ORANGE, weight="bold")

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Vehicle separation at transmission (m)")
    ax.set_xticks(frames)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    fit_note = "" if np.isnan(r) else f"  (separation fit r={r:.5f})"
    ax.set_title(
        f"Frame delivery vs. vehicle separation -- {args.seq} / {args.config}\n"
        f"delivery is a distance cliff, not a smooth decay{fit_note}",
        fontsize=13)

    fig.tight_layout()

    out = Path(args.out) if args.out else (
        Path("report/figures") / f"distance_vs_frames_{args.seq}_{args.config}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"source           : {src}")
    print(f"frames           : {len(rows)} total, {len(delivered)} delivered, "
          f"{len(lost)} lost")
    if not np.isnan(r):
        print(f"separation fit   : distance = {m:.3f} * tx_time + {b:.1f}  "
              f"(r={r:.5f})")
    if cliff is not None:
        print(f"observed cliff   : last delivered frame at {cliff:.1f} m")
        if lost_dist:
            print(f"                   first lost frame at ~{min(lost_dist):.1f} m "
                  f"(estimated)")
    print(f"wrote            : {out}")


if __name__ == "__main__":
    main()
