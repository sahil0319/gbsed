#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_decode.py -- receiver side of the pixel baseline.

Takes whatever the Veins receiver reassembled, decodes it as an image, runs
the SAME detector and the SAME scene-graph extraction the GBSED encoder used,
and scores the resulting graph against the SAME ground truth. The comparison
is therefore like-for-like: only the representation carried over the link
differs.

    received .bin -> cv2.imdecode -> upscale to 1280x720 -> Faster R-CNN
                  -> SceneGraph -> compare against meta["graph"]

The upscale matters and is not a thumb on the scale: the BEV homography and
the extractor's distance thresholds are calibrated for 1280x720, so a real
receiver would have to do exactly this before it could extract anything.

Output columns match gbsed_decode.py's fidelity.csv wherever they mean the
same thing, plus rx_resolution and psnr_db to quantify pixel degradation.

Usage
-----
    python tools/image_decode.py --received <dir> --meta image_data_seq1_webp \
        --out decoded_img_seq1_webp
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import gbsed_semantic as gs

WORK_W, WORK_H = 1280, 720

# Metrics live in gbsed_semantic so both arms compute them identically:
# gs.edge_metrics / gs.actor_edges / gs.risky_edges.


def detect_boxes_from_array(img_bgr, device="cpu", score_thresh=0.5):
    """gs.detect_boxes, but from an in-memory frame instead of a file."""
    import torch
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    model = gs._get_detector(device)
    with torch.no_grad():
        out = model([tensor.to(device)])[0]
    keep = out["scores"] >= score_thresh
    return (out["boxes"][keep].cpu(),
            [int(l) for l in out["labels"][keep].cpu()],
            (WORK_H, WORK_W),
            [float(s) for s in out["scores"][keep].cpu()])


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)


def find_received(received_dir, meta):
    for name in ("received_" + meta["bin"], meta["bin"]):
        p = received_dir / name
        if p.is_file():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Decode received image payloads and score the scene graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--received", required=True)
    ap.add_argument("--meta", required=True,
                    help="the image_data_* directory produced by image_encode.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--save-images", action="store_true",
                    help="write each received frame to <out>/rx/ for inspection")
    args = ap.parse_args()

    received_dir = Path(args.received).expanduser().resolve()
    meta_dir = Path(args.meta).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rxdir = outdir / "rx"
    if args.save_images:
        rxdir.mkdir(exist_ok=True)

    metas = sorted(meta_dir.glob("*.meta.json"))
    if not metas:
        raise SystemExit("no *.meta.json in %s" % meta_dir)

    cfg = gs.load_config(args.config)
    ae, bev = gs.make_autoencoder(cfg)

    manifest = {}
    mf = meta_dir / "manifest.json"
    if mf.is_file():
        manifest = json.loads(mf.read_text())

    print("received from    : %s" % received_dir)
    print("payload set      : %s (mode=%s, codec=%s)"
          % (meta_dir.name, manifest.get("mode", "?"), manifest.get("codec")))
    print("frames           : %d" % len(metas))
    print()

    hdr = "%-5s %-10s %8s %11s %8s %11s %11s %8s" % (
        "idx", "status", "bytes", "rx_res", "psnr_dB", "nodes o/r", "edges o/r", "edge F1")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for mp in metas:
        meta = json.loads(mp.read_text())
        idx = meta["frame"]
        row = {
            "frame": idx, "source_image": meta["source_image"],
            "sent_bytes": meta["n_bytes"], "gbsed_bytes": meta.get("gbsed_n_bytes", ""),
            "received_bytes": 0, "status": "LOST", "bit_exact": False,
            "tx_resolution": "x".join(str(v) for v in meta.get("resolution", [])),
            "rx_resolution": "", "psnr_db": "",
            "n_detections": 0,
            "nodes_orig": meta["graph"]["n_nodes"], "nodes_rec": 0,
            "edges_orig": meta["graph"]["n_edges"], "edges_rec": 0,
            "edges_common": 0, "nodes_match": False,
            "edge_precision": 0.0, "edge_recall": 0.0, "edge_f1": 0.0,
            "actor_edges_orig": len(gs.actor_edges(
                [tuple(e) for e in meta["graph"]["edges"]])),
            "actor_edges_rec": 0, "actor_edges_common": 0,
            "actor_edge_precision": 0.0, "actor_edge_recall": 0.0,
            "actor_edge_f1": 0.0,
            "risky_orig": len(gs.risky_edges(
                [tuple(e) for e in meta["graph"]["edges"]])),
            "risky_rec": 0, "risky_preserved": 0,
            "note": "",
        }

        path = find_received(received_dir, meta)
        if path is None:
            row["note"] = "never completed at the receiver"
            rows.append(row)
            print("%-5d %-10s %8s %11s %8s %11s %11s %8s"
                  % (idx, "LOST", "-", "-", "-", "-", "-", "-"))
            continue

        raw = path.read_bytes()
        row["received_bytes"] = len(raw)
        row["bit_exact"] = (gs.sha256_of(raw) == meta["sha256"])

        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            row["status"] = "CORRUPT"
            row["note"] = "payload is not a decodable image"
            rows.append(row)
            print("%-5d %-10s %8d %11s %8s %11s %11s %8s"
                  % (idx, "CORRUPT", len(raw), "-", "-", "-", "-", "-"))
            continue

        row["rx_resolution"] = "%dx%d" % (img.shape[1], img.shape[0])

        # A real receiver must bring the frame back to the calibrated size
        # before the BEV and the distance thresholds mean anything.
        work = cv2.resize(img, (WORK_W, WORK_H), interpolation=cv2.INTER_CUBIC)

        src = Path(meta["source_image"])
        if src.is_file():
            orig = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if orig is not None:
                orig = cv2.resize(orig, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
                row["psnr_db"] = round(psnr(orig, work), 2)

        if args.save_images:
            cv2.imwrite(str(rxdir / (meta["stem"] + "_received.png")), img)

        boxes, labels_, image_size, scores = detect_boxes_from_array(
            work, args.device, args.score_thresh)
        row["n_detections"] = len(labels_)

        sg = gs.scene_graph_from_boxes(cfg, bev, boxes, labels_, image_size)
        ok, lines, stats = gs.compare_summary(meta["graph"], sg)
        em = gs.edge_metrics(meta["graph"]["edges"], gs.edge_set(sg))
        n_rec, n_orig = stats["n_edges_rec"], stats["n_edges_orig"]
        f1 = em["edge_f1"]

        row.update({
            "status": "EXACT" if ok else "DEGRADED",
            "nodes_rec": stats["n_nodes_rec"], "edges_rec": n_rec,
            "edges_common": stats["n_edges_common"],
            "nodes_match": stats["nodes_match"],
        })
        row.update(em)
        if row["n_detections"] == 0:
            row["note"] = "detector found nothing in the received frame"

        print("%-5d %-10s %8d %11s %8s %11s %11s %8.3f"
              % (idx, row["status"], len(raw), row["rx_resolution"],
                 row["psnr_db"] if row["psnr_db"] != "" else "-",
                 "%d/%d" % (stats["n_nodes_orig"], stats["n_nodes_rec"]),
                 "%d/%d" % (n_orig, n_rec), f1))
        rows.append(row)

    csv_path = outdir / "fidelity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    delivered = [r for r in rows if r["status"] != "LOST"]
    exact = [r for r in rows if r["status"] == "EXACT"]
    f1s = [r["edge_f1"] for r in delivered]
    psnrs = [r["psnr_db"] for r in delivered if r["psnr_db"] != ""]

    print()
    print("=" * len(hdr))
    print("frames           : %d" % n)
    print("delivered        : %d (%.1f%%)" % (len(delivered), 100.0 * len(delivered) / n))
    print("graph exact      : %d (%.1f%%)" % (len(exact), 100.0 * len(exact) / n))
    print("mean edge F1     : %.3f | delivered, %.3f | all frames"
          % (sum(f1s) / len(f1s) if f1s else 0.0,
             sum(r["edge_f1"] for r in rows) / n))
    print("mean actor F1    : %.3f | delivered  (skeleton excluded)"
          % (sum(r["actor_edge_f1"] for r in delivered) / len(delivered)
             if delivered else 0.0))
    if psnrs:
        print("mean PSNR        : %.2f dB (delivered frames)" % (sum(psnrs) / len(psnrs)))
    print("detections       : %d total across delivered frames"
          % sum(r["n_detections"] for r in delivered))
    ro = sum(r["risky_orig"] for r in delivered)
    rp = sum(r["risky_preserved"] for r in delivered)
    print("safety relations : %d of %d preserved on delivered frames%s"
          % (rp, ro, "" if not ro else "  (%.1f%%)" % (100.0 * rp / ro)))
    print("fidelity.csv     : %s" % csv_path)


if __name__ == "__main__":
    main()
