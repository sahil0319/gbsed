#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
budget_sweep.py -- how many bytes does each representation need?

The channel comparison shows that at ONE budget (GBSED's own) pixels recover
nothing. That invites the obvious question: how much more bandwidth would
pixels need before they catch up? This sweeps the image byte budget upward
and measures where, if anywhere, that happens.

Deliberately channel-free. Delivery is held out of it so the curve measures
representation efficiency alone -- bytes in, semantic content out. Putting it
back only hurts the image arm further, because larger payloads span more
chunks and a single lost chunk loses the frame.

    for each budget B:  image -> encode to <=B bytes -> decode -> detector
                              -> SceneGraph -> score against ground truth

Two reference points anchor the curve:
  * GBSED at its native size (the graph payloads themselves, decoded).
  * The original 1280x720 frame, i.e. an unbounded budget. The ground truth
    was produced from exactly this input, so the image arm must reach 1.0
    there. If it does not, the image pipeline is broken rather than the
    representation being weak -- this is the sweep's own sanity check.

Usage:
    python tools/budget_sweep.py --meta scene_data_seq1 --images images_seq1 \
        --out budget_sweep
"""

import os
import sys
import csv
import json
import glob
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gbsed_semantic as gs
from image_encode import fit_to_budget, encode_at, WORK_W, WORK_H, SCALES, QUALITIES
from image_decode import detect_boxes_from_array

DEFAULT_SCALES = [1, 2, 4, 8, 16, 32, 64, 128]


def score_graph(cfg, bev, img_bgr, truth_graph, device, thresh):
    """Detector -> SceneGraph -> metrics against the ground-truth edge list."""
    boxes, labels_, image_size, _ = detect_boxes_from_array(img_bgr, device, thresh)
    sg = gs.scene_graph_from_boxes(cfg, bev, boxes, labels_, image_size)
    em = gs.edge_metrics(truth_graph["edges"], gs.edge_set(sg))
    em["n_detections"] = len(labels_)
    em["exact"] = (gs.node_list(sg) == [tuple(n) for n in truth_graph["nodes"]]
                   or [list(x) for x in gs.node_list(sg)] == truth_graph["nodes"]) \
                  and sorted(gs.edge_set(sg)) == sorted(tuple(e) for e in truth_graph["edges"])
    return em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="scene_data_seq1")
    ap.add_argument("--images", default="images_seq1")
    ap.add_argument("--out", default="budget_sweep")
    ap.add_argument("--codecs", nargs="+", default=["webp", "jpeg"])
    ap.add_argument("--scales", nargs="+", type=float, default=DEFAULT_SCALES)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    metas = [json.load(open(f)) for f in sorted(glob.glob(str(Path(args.meta) / "*.meta.json")))]
    if not metas:
        raise SystemExit("no metas in %s" % args.meta)
    img_dir = Path(args.images)

    cfg = gs.load_config(args.config)
    ae, bev = gs.make_autoencoder(cfg)

    # Cache the working-size frames once; the sweep re-encodes them many times.
    frames = {}
    for m in metas:
        src = img_dir / os.path.basename(m["source_image"])
        im = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if im is None:
            raise SystemExit("cannot read %s" % src)
        frames[m["frame"]] = cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

    gbsed_bytes = sum(m["n_bytes"] for m in metas) / len(metas)
    rows = []

    def aggregate(label, codec, scale, per_frame):
        tot = lambda k: sum(d[k] for d in per_frame)
        r = {
            "arm": label, "codec": codec, "budget_scale": scale,
            "mean_bytes": round(sum(d["bytes"] for d in per_frame) / len(per_frame), 1),
            "total_bytes": tot("bytes"),
            "frames": len(per_frame),
            "graph_exact": sum(1 for d in per_frame if d["exact"]),
            "mean_edge_f1": round(sum(d["edge_f1"] for d in per_frame) / len(per_frame), 4),
            "mean_actor_f1": round(sum(d["actor_edge_f1"] for d in per_frame) / len(per_frame), 4),
            "risky_total": tot("risky_orig"),
            "risky_preserved": tot("risky_preserved"),
            "detections": tot("n_detections"),
        }
        r["graph_exact_rate"] = round(r["graph_exact"] / r["frames"], 4)
        r["risky_recall"] = round(r["risky_preserved"] / r["risky_total"], 4) \
            if r["risky_total"] else 0.0
        r["bytes_per_relation"] = round(r["total_bytes"] / r["risky_preserved"], 1) \
            if r["risky_preserved"] else None
        return r

    # ---- reference: GBSED's own payloads, decoded ------------------------
    print(">> GBSED reference (native payload, decoded losslessly)")
    per = []
    for m in metas:
        raw = (Path(args.meta) / m["bin"]).read_bytes()
        sg, _ = gs.decode_payload(ae, raw)
        em = gs.edge_metrics(m["graph"]["edges"], gs.edge_set(sg))
        em["n_detections"] = len([n for n in gs.node_list(sg) if gs.is_actor(n[0])])
        em["exact"] = sorted(gs.edge_set(sg)) == sorted(tuple(e) for e in m["graph"]["edges"])
        em["bytes"] = m["n_bytes"]
        per.append(em)
    rows.append(aggregate("GBSED", "scene-graph", 1.0, per))
    print("   %d frames, %.0f B/frame, exact %d/%d, safety %d/%d"
          % (len(per), rows[-1]["mean_bytes"], rows[-1]["graph_exact"], len(per),
             rows[-1]["risky_preserved"], rows[-1]["risky_total"]))

    # ---- reference: the original frame, unbounded budget -----------------
    print(">> Image reference (original 1280x720, unbounded budget)")
    per = []
    for m in metas:
        em = score_graph(cfg, bev, frames[m["frame"]], m["graph"],
                         args.device, args.score_thresh)
        em["bytes"] = os.path.getsize(img_dir / os.path.basename(m["source_image"]))
        per.append(em)
    rows.append(aggregate("Image (original)", "none", float("inf"), per))
    print("   %.0f B/frame, exact %d/%d, safety %d/%d, actor F1 %.3f"
          % (rows[-1]["mean_bytes"], rows[-1]["graph_exact"], len(per),
             rows[-1]["risky_preserved"], rows[-1]["risky_total"],
             rows[-1]["mean_actor_f1"]))
    if rows[-1]["risky_recall"] < 0.99:
        print("   !! sanity check: the image arm does not reach 1.0 at unbounded")
        print("      budget, so the gap below is not purely a bitrate effect.")

    # ---- the sweep --------------------------------------------------------
    for codec in args.codecs:
        for scale in args.scales:
            per = []
            for m in metas:
                budget = int(round(m["n_bytes"] * scale))
                payload, dims, q = fit_to_budget(frames[m["frame"]], codec, budget)
                if payload is None:
                    payload, dims = encode_at(frames[m["frame"]], codec,
                                              SCALES[-1], QUALITIES[-1])
                dec = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                work = cv2.resize(dec, (WORK_W, WORK_H), interpolation=cv2.INTER_CUBIC)
                em = score_graph(cfg, bev, work, m["graph"], args.device, args.score_thresh)
                em["bytes"] = len(payload)
                per.append(em)
            r = aggregate("Image (%s)" % codec, codec, scale, per)
            rows.append(r)
            print(">> %-14s x%-5g  %7.0f B/frame  exact %2d/%d  safety %2d/%-3d "
                  "actorF1 %.3f  det %d"
                  % (codec, scale, r["mean_bytes"], r["graph_exact"], r["frames"],
                     r["risky_preserved"], r["risky_total"], r["mean_actor_f1"],
                     r["detections"]))

    cols = ["arm", "codec", "budget_scale", "mean_bytes", "total_bytes", "frames",
            "graph_exact", "graph_exact_rate", "mean_edge_f1", "mean_actor_f1",
            "risky_total", "risky_preserved", "risky_recall", "detections",
            "bytes_per_relation"]
    with open(outdir / "budget_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # ------------------------------------------------------------------ plot
    gb = rows[0]
    orig = rows[1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = {"webp": "#d6604d", "jpeg": "#8c6bb1"}

    for ax, key, title, ylab in [
        (axes[0], "risky_recall", "Safety-critical relations preserved",
         "fraction of near_coll / super_near with ego"),
        (axes[1], "mean_actor_f1", "Actor-relation F1  (skeleton excluded)",
         "mean F1 over edges involving a detected actor"),
    ]:
        for codec in args.codecs:
            pts = [r for r in rows if r["codec"] == codec]
            pts.sort(key=lambda r: r["mean_bytes"])
            ax.plot([p["mean_bytes"] for p in pts], [p[key] for p in pts],
                    marker="o", color=colors.get(codec, "#777"), lw=2, ms=6,
                    label="Image (%s)" % codec.upper())
        ax.scatter([gb["mean_bytes"]], [gb[key]], s=200, marker="*",
                   color="#1b7837", zorder=5, label="GBSED (scene graph)")
        ax.axhline(orig[key], ls=":", color="#444", lw=1.2)
        ax.text(ax.get_xlim()[1], orig[key], " original image\n (unbounded)",
                va="center", fontsize=7.5, color="#444")
        ax.axvline(gb["mean_bytes"], ls="--", color="#1b7837", lw=1, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("bytes per frame (log)")
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="center right")

    fig.suptitle("Semantic content recovered per byte spent — %d frames, no channel"
                 % gb["frames"], fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "01_rate_semantics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # how much more budget do pixels need to match GBSED?
    print()
    for codec in args.codecs:
        pts = sorted([r for r in rows if r["codec"] == codec],
                     key=lambda r: r["mean_bytes"])
        hit = next((p for p in pts if p["risky_recall"] >= gb["risky_recall"]), None)
        if hit:
            print("%s reaches GBSED's safety recall at %.0f B/frame -- %.0fx GBSED"
                  % (codec.upper(), hit["mean_bytes"], hit["mean_bytes"] / gb["mean_bytes"]))
        else:
            top = pts[-1]
            print("%s never reaches GBSED's safety recall within the sweep "
                  "(best %.2f at %.0f B/frame = %.0fx GBSED)"
                  % (codec.upper(), top["risky_recall"], top["mean_bytes"],
                     top["mean_bytes"] / gb["mean_bytes"]))
    print("\nwrote %s and %s" % (outdir / "budget_sweep.csv",
                                 outdir / "01_rate_semantics.png"))


if __name__ == "__main__":
    main()
