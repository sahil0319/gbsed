#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_slide_figures.py -- regenerate the presentation figures for the results
slides. Presentation-tuned: large fonts, thick lines, legible when projected.

Produces five figures into report/slides_figures/, one per results slide:

    network_delivery.png   slide "Results: Network Delivery"
    matched_budget.png     slide "Results: GBSED vs. Image Transmission"
    bandwidth_gap.png      slide "Results: How Much Bandwidth Do Images Need?"
    payload_size.png       slide "Results: Full-Frame Transmission Is Infeasible"
    loss_resilience.png    slide "Results: Graceful Semantic Degradation"

Every figure is built from committed experiment outputs (see
report/slides_figures/README.md for the exact source file behind each one).
No internal payload-format version numbers appear in any figure; the
slice-aligned packing is simply "the payload".

Usage:
    python tools/make_slide_figures.py
"""

import os
import sys
import csv
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "report" / "slides_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Presentation palette
GREEN = "#1b7837"     # GBSED / semantic
RED = "#d6604d"       # WebP
PURPLE = "#8c6bb1"    # JPEG
GREY = "#555555"
ORANGE = "#d95f02"

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
def fig_network_delivery():
    """Slide 14. Delivery rate collapsing with the noise floor, both sequences,
    from the GBSED channel sweep."""
    rows = list(csv.DictReader(open(REPO / "comparison" / "comparison.csv")))
    order = ["Baseline", "Noise_Low", "Noise_Medium", "Noise_High", "CAV_Extreme"]
    nf = {"Baseline": -98, "Noise_Low": -95, "Noise_Medium": -90,
          "Noise_High": -85, "CAV_Extreme": -80}

    gbsed = {r["config"]: float(r["delivery_rate"]) * 100
             for r in rows if "GBSED" in r["arm"]}
    seq1 = [gbsed.get(c, np.nan) for c in order]
    # seq2 delivery from the raw matrix (same five noise floors)
    m2 = {}
    for r in csv.DictReader(open(REPO / "experiment_results" / "results_matrix.csv")):
        if r["sequence"] == "seq2":
            key = {"CAV_Bad": "Noise_High"}.get(r["config"], r["config"])
            m2.setdefault(r["config"], float(r["delivery_rate"]) * 100)
    seq2map = {"Baseline": "Baseline", "Noise_Low": "Noise_Low",
               "Noise_Medium": "Noise_Medium", "Noise_High": "Noise_High"}
    seq2 = [m2.get(seq2map.get(c, c), np.nan) for c in order]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(x, seq1, "o-", color=GREEN, lw=3, ms=11, label="Sequence 1 (20 frames)")
    ax.plot(x, seq2, "s--", color=ORANGE, lw=3, ms=10, label="Sequence 2 (23 frames)")
    for xi, yi in zip(x, seq1):
        if not np.isnan(yi):
            ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=12, color=GREEN,
                        weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{nf[c]} dBm" for c in order], fontsize=12)
    ax.set_ylabel("Frames successfully delivered (%)")
    ax.set_ylim(-5, 100)
    ax.set_xlabel("Channel condition (rising noise floor →)")
    ax.set_title("Delivery collapses as the channel degrades\n"
                 "every delivered frame reconstructs exactly", fontsize=15)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "network_delivery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote network_delivery.png")


def fig_matched_budget():
    """Slide "Results: same byte budget, real channel". Redesigned from the
    original two-disconnected-bar-charts version, which readers found
    confusing ("what does 70% delivered even mean here?"). The new version
    makes the causal chain explicit and shows one concrete frame's ACTUAL
    received bytes, decoded, next to the ACTUAL reconstructed scene graph --
    not just aggregate bars -- so "0 safety relations" is something you can
    see rather than only a number.

    Uses frame_0013 (00097109.jpg): the densest frame in the matched-budget
    set, GBSED payload 1354 B, and it contains a near_coll (collision-risk)
    relation, so the contrast is as concrete as possible. This frame was
    actually delivered under Baseline in all three arms -- the images shown
    are the real received-and-decoded bytes, not illustrations.

    Every element is placed with an explicit figure-fraction rectangle
    (fig.add_axes) rather than gridspec + manual overrides, to avoid the
    title/panel collisions that approach produced.
    """
    import json
    import gbsed_semantic as gs
    import matplotlib.image as mpimg
    from matplotlib.patches import FancyBboxPatch
    import cv2

    stem = "frame_0013"
    gbsed_meta_path = REPO / "scene_data_seq1" / f"{stem}.meta.json"
    if not gbsed_meta_path.is_file():
        print("SKIP fig_matched_budget: %s not found" % gbsed_meta_path)
        return

    cfg = gs.load_config()
    ae, _ = gs.make_autoencoder(cfg)
    raw = (REPO / "scene_data_seq1" / f"{stem}.bin").read_bytes()
    sg, _ = gs.decode_payload(ae, raw)
    tmp = OUT / "_tmp_graph"
    tmp.mkdir(exist_ok=True)
    gs.maybe_visualize(sg, tmp / "graph.png")
    graph_img = mpimg.imread(tmp / "graph.png")

    def load_received(setname):
        p = (REPO / "experiment_results_image" /
             f"received_image_data_seq1_{setname}_Baseline" / f"received_{stem}.bin")
        raw = p.read_bytes()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, len(raw)

    webp_img, webp_bytes = load_received("webp")
    jpeg_img, jpeg_bytes = load_received("jpeg")
    gbsed_bytes = len(raw)

    # -------------------------------------------------------------- layout
    # All positions are explicit [left, bottom, width, height] in figure
    # fraction (0-1), top to bottom, with a fixed gap between every band so
    # nothing can overlap regardless of image aspect ratio.
    fig = plt.figure(figsize=(15, 10.2))

    TITLE_Y   = 0.965
    BAND1_TOP, BAND1_BOT   = 0.86, 0.70    # the 4 causal-chain steps
    BAND2_TOP, BAND2_BOT   = 0.66, 0.28    # the 3 real outputs for one frame
    BAND2_IMG_TOP, BAND2_IMG_BOT = 0.62, 0.34   # image area within band 2
    BAND3_TOP, BAND3_BOT   = 0.20, 0.06    # aggregate bar chart

    fig.text(0.5, TITLE_Y,
             "Same byte budget, same real channel: only the meaning of the bytes differs",
             ha="center", fontsize=16, weight="bold")

    # ---- band 1: the causal chain, one box per step -----------------------
    axT = fig.add_axes([0.03, BAND1_BOT, 0.94, BAND1_TOP - BAND1_BOT])
    axT.axis("off")
    steps = [
        ("1", f"Each encoding is given the\nSAME byte budget\n(~{gbsed_bytes} B/frame)"),
        ("2", "All three are sent over the\nSAME live 802.11p simulation\n(Baseline channel)"),
        ("3", "Delivery is IDENTICAL: 70%\n(14/20 frames) -- the channel\nonly sees packet size"),
        ("4", "But what those bytes DECODE\nINTO is not identical --\nsee below"),
    ]
    n = len(steps)
    gap = 0.03
    w = (1 - gap * (n - 1)) / n
    for i, (num, txt) in enumerate(steps):
        x0 = i * (w + gap)
        col = GREEN if i < 3 else ORANGE
        axT.add_patch(FancyBboxPatch((x0, 0.06), w, 0.88,
            boxstyle="round,pad=0.02", facecolor="#f5f5f5", edgecolor=col,
            linewidth=1.8, transform=axT.transAxes, clip_on=False))
        axT.text(x0 + w / 2, 0.80, num, ha="center", va="top", fontsize=16,
                 weight="bold", color=col, transform=axT.transAxes)
        axT.text(x0 + w / 2, 0.62, txt, ha="center", va="top", fontsize=10.8,
                 transform=axT.transAxes)
        if i < n - 1:
            axT.annotate("", xy=(x0 + w + gap * 0.15, 0.5),
                        xytext=(x0 + w - gap * 0.15 + gap, 0.5),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.8))
    axT.set_xlim(0, 1); axT.set_ylim(0, 1)

    # ---- band 2: one real frame, three real outcomes -----------------------
    # Ground truth for this frame: 18/18 edges recovered exactly by GBSED
    # (experiment_results/decoded_seq1_Baseline/fidelity.csv, frame 13,
    # status=EXACT), including 2 near_coll (collision-risk) instances.
    # WebP's detector actually found 2 objects in the received image but
    # recovered neither near_coll relation; JPEG's detector found nothing.
    # (experiment_results_image/decoded_image_data_seq1_{webp,jpeg}_Baseline/
    # fidelity.csv, frame 13: n_detections 2 and 0 respectively.) Both
    # numbers are used as measured, not asserted -- "2 detected, still 0/2"
    # is a stronger and more honest claim than "nothing detectable".
    panels = [
        (graph_img, GREEN,
         f"GBSED scene graph\n{gbsed_bytes} B received",
         "18/18 edges recovered exactly\nnear_coll PRESERVED (2/2)", True),
        (webp_img, RED,
         f"WebP image\n{webp_bytes} B received, {webp_img.shape[1]}×{webp_img.shape[0]} px",
         "2 objects detected, but\n0/2 near_coll relations recovered", False),
        (jpeg_img, PURPLE,
         f"JPEG image\n{jpeg_bytes} B received, {jpeg_img.shape[1]}×{jpeg_img.shape[0]} px",
         "0 objects detected --\n0/2 near_coll relations recovered", False),
    ]
    pw = 0.29
    px = [0.035, 0.355, 0.675]
    for (img, col, title, verdict, ok), x0 in zip(panels, px):
        fig.text(x0 + pw / 2, BAND2_TOP + 0.005, title, ha="center", va="bottom",
                 fontsize=11.5, weight="bold", color=col)
        ax = fig.add_axes([x0, BAND2_IMG_BOT, pw, BAND2_IMG_TOP - BAND2_IMG_BOT])
        ax.imshow(img, interpolation="nearest" if not ok else None)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2.4)
        vcol = GREEN if ok else RED
        fig.text(x0 + pw / 2, BAND2_IMG_BOT - 0.015, verdict, ha="center",
                 va="top", fontsize=11, weight="bold", color=vcol)

    # ---- band 3: the aggregate result, all 20 frames -----------------------
    axB = fig.add_axes([0.12, BAND3_BOT, 0.80, BAND3_TOP - BAND3_BOT])
    arms = ["GBSED", "WebP", "JPEG"]
    colors = [GREEN, RED, PURPLE]
    safety = [32, 0, 0]
    bars = axB.barh(arms, safety, color=colors, height=0.55)
    for bar, v in zip(bars, safety):
        axB.text(v + 0.6, bar.get_y() + bar.get_height() / 2, f"{v}/32",
                 va="center", fontsize=12.5, weight="bold",
                 color=(GREEN if v else RED))
    axB.set_xlim(0, 36)
    axB.set_xlabel("Safety-critical relations preserved, all 20 frames "
                   "(identical delivery in all three arms)", fontsize=10.5)
    axB.invert_yaxis()
    axB.grid(alpha=0.3, axis="x")
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    out = OUT / "matched_budget.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    print("wrote matched_budget.png")


def fig_bandwidth_gap():
    """Slide 16. Rate-semantics curve: how far right the image codecs must
    travel to reach GBSED's safety-relation recall."""
    rows = list(csv.DictReader(open(REPO / "budget_sweep" / "budget_sweep.csv")))
    def series(codec):
        pts = [(float(r["mean_bytes"]), float(r["risky_recall"]))
               for r in rows if r["codec"] == codec]
        pts.sort()
        return zip(*pts)

    gb = next(r for r in rows if r["arm"] == "GBSED")
    gbx, gby = float(gb["mean_bytes"]), float(gb["risky_recall"])

    fig, ax = plt.subplots(figsize=(11, 6))
    wx, wy = series("webp")
    jx, jy = series("jpeg")
    ax.plot(wx, wy, "o-", color=RED, lw=2.8, ms=8, label="Image (WebP)")
    ax.plot(jx, jy, "s-", color=PURPLE, lw=2.8, ms=8, label="Image (JPEG)")
    ax.scatter([gbx], [gby], s=520, marker="*", color=GREEN, zorder=6,
               edgecolors="black", linewidths=0.6, label="GBSED (scene graph)")

    # the 96x annotation
    match = 77809
    ax.axvline(gbx, ls="--", color=GREEN, lw=1.4, alpha=0.6)
    ax.annotate("GBSED\n811 B", (gbx, 0.30), color=GREEN, fontsize=13,
                weight="bold", ha="center")
    ax.annotate("", xy=(match, 0.5), xytext=(gbx, 0.5),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.8))
    ax.text(np.sqrt(gbx * match), 0.545, "≈ 96× more bandwidth",
            ha="center", fontsize=13, color=GREY, weight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Bytes per frame (log scale)")
    ax.set_ylabel("Safety-critical relations preserved")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Images need far more bandwidth to carry the same safety content",
                 fontsize=15)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "bandwidth_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote bandwidth_gap.png")


def fig_payload_size():
    """Slide 17. Payload size, log scale, GBSED vs matched image vs full frame."""
    labels = ["GBSED\nscene graph", "Image\n(matched budget)", "Full JPEG\nframe"]
    vals = [16218, 14604, 24839399]
    colors = [GREEN, RED, GREY]
    base = vals[0]
    full_mult = int(vals[2] / base)            # 1531x, matching the report and slides
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Total bytes for 20 frames (log scale)")
    labelmults = ["1×", "same budget", f"{full_mult:,}×"]
    for bar, v, ml in zip(bars, vals, labelmults):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.4,
                f"{v:,} B\n({ml})", ha="center", fontsize=13, weight="bold")
    ax.set_ylim(top=max(vals) * 12)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"Sending whole frames costs {full_mult:,}× the bandwidth\n"
                 "— more than the vehicles' contact window allows",
                 fontsize=14.5)
    fig.tight_layout()
    fig.savefig(OUT / "payload_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote payload_size.png")


def fig_loss_resilience():
    """Slide 18. One frame, one lost chunk: the safety-critical relation
    survives because it rode in the delivered chunk. Renders the two scene
    graphs from the actual sent and received payloads."""
    import gbsed_semantic as gs
    import matplotlib.image as mpimg
    from matplotlib.patches import FancyBboxPatch

    cfg = gs.load_config()
    ae, _ = gs.make_autoencoder(cfg)

    meta = json.load(open(REPO / "scene_data_seq1_v2_cs500" / "frame_0005.meta.json"))
    full_path = REPO / "scene_data_seq1_v2_cs500" / "frame_0005.bin"
    rx_path = (REPO / "experiment_results_format" /
               "received_v2_cs500_Noise_Low" / "received_frame_0005.bin")
    if not rx_path.is_file():
        print("SKIP loss_resilience: partial-delivery output not found at\n  %s"
              "\n  (run ./tools/run_format_sweep.sh first)" % rx_path)
        return

    tmp = OUT / "_graphs"
    tmp.mkdir(exist_ok=True)
    renders = {}
    for key, (path, cs, fmt) in {
        "full": (full_path, 500, "v2"),
        "recv": (rx_path, 500, "v2"),
    }.items():
        sg, _ = gs.decode_payload(ae, open(path, "rb").read(), cs, fmt)
        gs.maybe_visualize(sg, tmp / f"{key}.png")
        renders[key] = tmp / f"{key}.png"

    fig = plt.figure(figsize=(15.5, 7.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.27], hspace=0.04, wspace=0.06)
    panels = [
        ("full", "(a)   TRANSMITTED",
         "7 nodes  ·  16 edges  ·  7 relation types",
         "isIn · inDFrontOf · atDRearOf · toLeftOf\n"
         "toRightOf · super_near · very_near"),
        ("recv", "(b)   RECEIVED  —  one of two chunks lost",
         "7 nodes  ·  10 edges  ·  3 relation types",
         "kept:  isIn · super_near · very_near\n"
         "lost:  inDFrontOf · atDRearOf · toLeftOf · toRightOf"),
    ]
    for i, (key, title, stats, body) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(mpimg.imread(renders[key]))
        ax.set_anchor("N"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(GREEN); sp.set_linewidth(2)
        ax.set_title(title, fontsize=15, weight="bold", color=GREEN, pad=11)

        ax2 = fig.add_subplot(grid[1, i]); ax2.axis("off")
        ax2.add_patch(FancyBboxPatch((0.02, 0.06), 0.96, 0.88,
            boxstyle="round,pad=0.02", facecolor="#f2f9f3", edgecolor=GREEN,
            linewidth=1.5, transform=ax2.transAxes, clip_on=False))
        ax2.text(0.5, 0.82, stats, ha="center", va="top", fontsize=13,
                 transform=ax2.transAxes)
        ax2.text(0.5, 0.52, body, ha="center", va="top", fontsize=11,
                 family="monospace", color="#333", transform=ax2.transAxes)
        if i == 1:
            ax2.text(0.5, 0.15, "collision-risk relation  super_near  —  "
                     "PRESERVED  (2 of 2)", ha="center", va="top", fontsize=13,
                     weight="bold", color=ORANGE, transform=ax2.transAxes)

    fig.suptitle("A scene graph survives packet loss: the safety-critical "
                 "relation is preserved", fontsize=16.5, weight="bold", y=1.03)
    fig.text(0.5, 0.965, "One frame, 1000 bytes in two chunks. The first chunk "
             "was heard at 359 m; the second never arrived.",
             ha="center", fontsize=12.5, color="#444")
    fig.text(0.5, 0.01, "The payload carries whole relation slices, collision-risk "
             "relations first, so a lost chunk costs directional context — where "
             "vehicles are\nrelative to the ego car — while the relations a braking "
             "decision depends on ride in the first chunk and survive intact.",
             ha="center", fontsize=11, color="#222", style="italic")
    fig.savefig(OUT / "loss_resilience.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    # tidy the intermediate renders
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()
    print("wrote loss_resilience.png")


if __name__ == "__main__":
    fig_network_delivery()
    fig_matched_budget()
    fig_bandwidth_gap()
    fig_payload_size()
    fig_loss_resilience()
    print("\nAll figures written to %s" % OUT)
