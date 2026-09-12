#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GBSED encoder -- a folder of driving images to a folder of transmittable .bin files.

    image -> SceneGraph -> encode() -> sem_compression() -> format_storage() -> .bin

Each frame produces two files:

    frame_0000.bin        the payload, exactly what pipeline.GBSED would have
                          handed the MIMO-OFDM model.  This is what Veins sends.
    frame_0000.meta.json  ground truth -- node list, edge list, payload hash.
                          NOT transmitted; the decoder scores against it.

Plus one `manifest.json` for the whole run, which carries the semicolon-joined
path string to paste into omnetpp.ini.

Example
-------
    python tools/gbsed_encode.py --images /path/to/images --out scene_data \
        --stage /path/to/gbsed_veins/src/veins/modules/application/gbsed/GBSEDApp

Set GBSED_VEINS_APP to that scenario directory once and --stage can be omitted;
see RUNNING.md in the Veins repository.
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import gbsed_semantic as gs

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# How many chunks the sender can physically transmit in one run:
#     (vehicle_lifetime - startTime) / sendInterval
# startTime and sendInterval are NED parameters in omnetpp.ini; the vehicle
# lifetime follows from the route length and maxSpeed in gbsed.rou.xml.  The
# default below matches the scenario shipped with the Veins repo (vehicles
# live ~125 s, startTime 10 s, sendInterval 2.5 s).  Override with --tx-budget
# if your scenario differs.
VEINS_TX_BUDGET = 45


def collect_images(patterns):
    """Accept directories, globs, and plain files; return a sorted unique list."""
    found = []
    for pat in patterns:
        p = Path(pat).expanduser()
        if p.is_dir():
            found += [q for q in p.iterdir()
                      if q.is_file() and q.suffix.lower() in IMAGE_EXTS]
        elif p.is_file():
            found.append(p)
        else:                                        # treat as a glob
            base = Path(p.anchor or ".")
            rel = str(p.relative_to(p.anchor)) if p.anchor else str(p)
            matches = [q for q in base.glob(rel) if q.is_file()]
            if not matches:
                print("warning: no files matched %r" % pat, file=sys.stderr)
            found += matches
    uniq = sorted({q.resolve() for q in found})
    return uniq


def main():
    ap = argparse.ArgumentParser(
        description="Encode driving images into GBSED semantic payloads (.bin).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True,
                    help="image folder(s), glob(s), or individual image file(s)")
    ap.add_argument("--out", default="scene_data",
                    help="output directory for .bin and .meta.json files")
    ap.add_argument("--config", default=None,
                    help="path to pipeline_extraction.yaml")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                    help="torch device for the detector")
    ap.add_argument("--score-thresh", type=float, default=0.5,
                    help="detector confidence threshold")
    ap.add_argument("--limit", type=int, default=None,
                    help="encode at most this many images")
    ap.add_argument("--stage", default=os.environ.get("GBSED_VEINS_APP"),
                    metavar="SCENARIO_DIR",
                    help="also copy the .bin files into SCENARIO_DIR/scene_data "
                         "(defaults to $GBSED_VEINS_APP)")
    ap.add_argument("--visualize", action="store_true",
                    help="render each source scene graph to <out>/png/")
    ap.add_argument("--keep-empty", action="store_true",
                    help="keep frames whose graph has only the road/ego/lane skeleton")
    ap.add_argument("--send-interval", type=float, default=2.5,
                    help="appl.sendInterval in the scenario, seconds; only "
                         "used to report when the last chunk goes out")
    ap.add_argument("--start-time", type=float, default=10.0,
                    help="appl.startTime in the scenario, seconds")
    ap.add_argument("--format", choices=["v1", "v2"], default="v1",
                    help="v1 reproduces pipeline.GBSED._format_storage_ byte for "
                         "byte; v2 aligns chunk boundaries with relation slices "
                         "so a lost chunk costs relation types, not the frame")
    ap.add_argument("--chunk-size", type=int, default=1000,
                    help="must match appl.chunkSize in omnetpp.ini; v2 pads "
                         "each block to exactly this size")
    ap.add_argument("--tx-budget", type=int, default=VEINS_TX_BUDGET,
                    help="chunks the scenario can transmit in one run; only "
                         "affects the warning printed at the end")
    args = ap.parse_args()

    images = collect_images(args.images)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit("No images found.")

    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    pngdir = outdir / "png"
    if args.visualize:
        pngdir.mkdir(exist_ok=True)

    cfg = gs.load_config(args.config)
    ae, bev = gs.make_autoencoder(cfg)
    fingerprint = gs.codebook_fingerprint(cfg)

    print("config           : %s" % cfg.yaml_path)
    print("bev calibration  : %s" % cfg.image_settings["BEV_PATH"])
    print("codebook         : %s  (%d actors, %d relations)"
          % (fingerprint,
             len(cfg.relation_extraction_settings["ACTOR_NAMES"]),
             len(cfg.relation_extraction_settings["RELATION_NAMES"])))
    print("detector         : torchvision fasterrcnn_resnet50_fpn on %s (thresh %.2f)"
          % (args.device, args.score_thresh))
    print("images           : %d" % len(images))
    print("output           : %s" % outdir)
    print()

    entries, skipped = [], []
    hdr = "%-5s %-28s %5s %6s %6s %8s  %s" % (
        "idx", "image", "nodes", "edges", "rels", "bytes", "detections")
    print(hdr)
    print("-" * len(hdr))

    for i, img in enumerate(images):
        t0 = time.time()
        try:
            boxes, labels_, image_size, scores = gs.detect_boxes(
                img, args.device, args.score_thresh)
        except SystemExit as e:
            print("  skipped %s (%s)" % (img.name, e))
            skipped.append({"image": str(img), "reason": "unreadable"})
            continue

        sg = gs.scene_graph_from_boxes(cfg, bev, boxes, labels_, image_size)

        # road + ego + 3 lanes = 5 nodes with no traffic participants at all
        if sg.g.number_of_nodes() <= 5 and not args.keep_empty:
            print("%-5d %-28s %5s %6s %6s %8s  %s"
                  % (i, img.name[:28], "-", "-", "-", "skip",
                     "no actors above threshold"))
            skipped.append({"image": str(img), "reason": "skeleton_only"})
            continue

        try:
            raw, detail = gs.encode_scene_graph(ae, sg, args.format, args.chunk_size)
        except ValueError as e:
            print("%-5d %-28s %5s %6s %6s %8s  %s"
                  % (i, img.name[:28], sg.g.number_of_nodes(), "-", "-", "skip", e))
            skipped.append({"image": str(img), "reason": "no_relations"})
            continue

        idx = len(entries)
        stem = "frame_%04d" % idx
        binpath = outdir / (stem + ".bin")
        binpath.write_bytes(raw)

        summary = gs.graph_summary(sg)
        meta = {
            "frame": idx,
            "stem": stem,
            "bin": binpath.name,
            "source_image": str(img),
            "codebook": fingerprint,
            "config": str(cfg.yaml_path),
            "n_bytes": len(raw),
            "format": args.format,
            "chunk_size": args.chunk_size,
            "v2": detail.get("v2"),
            "v1_equivalent_bytes": detail.get("v1_equivalent_bytes"),
            "n_float16": detail["n_float16"],
            "sha256": gs.sha256_of(raw),
            "labels": [int(v) for v in detail["labels"]],
            "actor_names": [cfg.relation_extraction_settings["ACTOR_NAMES"][v]
                            for v in detail["labels"]],
            "active_relation_indexes": detail["indexes"],
            "active_relation_names": [ae.rels[v] for v in detail["indexes"]],
            "T_shape": detail["T_shape"],
            "compT_shape": list(detail["compressed_Tensor"].shape),
            "feature_shape": list(detail["feature_nodes_matrix"].shape),
            "compression_ratio": round(
                float(detail["compressed_Tensor"].size)
                / max(int(np.prod(detail["T_shape"])), 1), 4),
            "detections": [
                {"coco_id": int(l), "coco_name": gs.COCO_CLASS_NAMES[int(l)],
                 "score": round(float(s), 3), "box": [round(float(v), 1) for v in b]}
                for b, l, s in zip(boxes.tolist(), labels_, scores)
            ],
            "graph": summary,
            "encode_seconds": round(time.time() - t0, 3),
        }
        (outdir / (stem + ".meta.json")).write_text(json.dumps(meta, indent=2))

        if args.visualize:
            gs.maybe_visualize(sg, pngdir / (stem + "_original.png"))

        det_names = [gs.COCO_CLASS_NAMES[int(l)] for l in labels_]
        print("%-5d %-28s %5d %6d %6d %8d  %s"
              % (idx, img.name[:28], summary["n_nodes"], summary["n_edges"],
                 len(detail["indexes"]), len(raw),
                 ", ".join(det_names[:4]) + (" +%d" % (len(det_names) - 4)
                                             if len(det_names) > 4 else "")))
        entries.append(meta)

    if not entries:
        raise SystemExit("\nNothing encoded. Try lowering --score-thresh, or "
                         "--keep-empty to keep skeleton-only frames.")

    # ---- staging into the Veins scenario -------------------------------
    staged_dir = None
    if args.stage:
        staged_dir = Path(args.stage).expanduser().resolve() / "scene_data"
        staged_dir.mkdir(parents=True, exist_ok=True)
        for e in entries:
            shutil.copy2(outdir / e["bin"], staged_dir / e["bin"])

    # The sender takes a ';'-separated queue; paths are relative to the
    # scenario directory, which is the simulation's working directory.
    file_path_param = ";".join("scene_data/" + e["bin"] for e in entries)

    total_bytes = sum(e["n_bytes"] for e in entries)
    manifest = {
        "codebook": fingerprint,
        "config": str(cfg.yaml_path),
        "n_frames": len(entries),
        "n_skipped": len(skipped),
        "total_bytes": total_bytes,
        "frames": [{"frame": e["frame"], "bin": e["bin"],
                    "source_image": e["source_image"],
                    "n_bytes": e["n_bytes"], "sha256": e["sha256"]}
                   for e in entries],
        "skipped": skipped,
        "omnetpp_filePath": file_path_param,
        "staged_to": str(staged_dir) if staged_dir else None,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print("encoded          : %d frame(s), %d skipped" % (len(entries), len(skipped)))
    if args.format == "v2":
        pad = sum(e["v2"]["padding_bytes"] for e in entries if e.get("v2"))
        v1b = sum(e["v1_equivalent_bytes"] for e in entries)
        print("format           : v2 slice-aligned, %d B blocks "
              "(padding %d B = %.0f%%; v1 would be %d B)"
              % (args.chunk_size, pad, 100.0 * pad / max(total_bytes, 1), v1b))
    print("total payload    : %d bytes (mean %.0f B/frame, max %d B)"
          % (total_bytes, total_bytes / len(entries),
             max(e["n_bytes"] for e in entries)))
    print("manifest         : %s" % (outdir / "manifest.json"))
    if staged_dir:
        print("staged to        : %s" % staged_dir)

    # ---- transmission budget -------------------------------------------
    chunk = args.chunk_size
    n_chunks = sum(-(-e["n_bytes"] // chunk) for e in entries)
    # Exactly one chunk per send event: the event that finishes a file also
    # loads the next one and reschedules at +2 s, so a file switch costs
    # nothing extra (GBSEDApp.cc handleSelfMsg).
    n_events = n_chunks
    print()
    print("transmission     : %d chunk(s) @%d B over %d frame(s) -> %d send events"
          % (n_chunks, chunk, len(entries), n_events))
    print("                   first at t=%g s, one every %g s, last at t=%g s"
          % (args.start_time, args.send_interval,
             args.start_time + args.send_interval * (n_events - 1)))
    if n_events > args.tx_budget:
        print("  WARNING: this scenario fits about %d send events." % args.tx_budget)
        print("  Chunks past that never leave node[0]. Either encode fewer "
              "frames (--limit), lower appl.sendInterval, or give the vehicles "
              "a longer route / lower maxSpeed in gbsed.rou.xml.")
    else:
        print("  fits the scenario's ~%d send-event budget." % args.tx_budget)

    print()
    print("omnetpp.ini:")
    print('  *.node[0].appl.filePath = "%s"' % file_path_param)


if __name__ == "__main__":
    main()
