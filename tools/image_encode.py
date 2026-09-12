#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_encode.py -- the pixel-domain baseline GBSED is compared against.

Produces .bin payloads carrying *images* rather than scene graphs, in the
same on-disk format the Veins harness already transmits, so the identical
simulation and identical channel configs can carry either one.

Two modes, answering two different questions:

  --mode full
      Transmit the source image as-is. Answers "what happens if you just
      send the camera frame?" Payloads are ~1.3 MB, i.e. ~1600x the GBSED
      payload, so this is about feasibility, not fidelity.

  --mode matched
      Re-encode each image down to the SAME byte budget as the GBSED payload
      for that frame, then transmit that. This is the controlled comparison:
      identical bytes on the wire, identical chunk count, therefore identical
      delivery under any given channel config. The only thing that differs is
      what those bytes *mean*. Anything the two approaches do differently
      here is a property of the representation, not of the link.

Frame numbering and ordering are taken from the GBSED run's *.meta.json, so
frame_0007 is the same source image in both experiments.

Usage
-----
    python tools/image_encode.py --meta scene_data_seq1 --images images_seq1 \
        --mode matched --codec jpeg --out image_data_seq1_matched
"""

import os
import sys
import json
import glob
import time
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

# The detector and the BEV calibration both assume this frame size; the
# encoder resizes to it before detection, so the baseline must start from the
# same pixels to be comparable.
WORK_W, WORK_H = 1280, 720

# Search grid for --mode matched. Largest image that fits the budget wins:
# scale is tried outer-to-inner so we keep as much resolution as possible,
# and quality is only dropped as far as needed at that scale.
SCALES = [1.0, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.125, 0.1,
          0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01]
QUALITIES = [90, 80, 70, 60, 50, 40, 30, 25, 20, 15, 10, 5, 1]


def encode_at(img, codec, scale, quality):
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if codec == "jpeg":
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    elif codec == "webp":
        ok, buf = cv2.imencode(".webp", small, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
    else:
        raise ValueError("unknown codec %r" % codec)
    if not ok:
        return None, (nw, nh)
    return buf.tobytes(), (nw, nh)


def fit_to_budget(img, codec, target_bytes):
    """Largest/highest-quality encoding of `img` that fits in target_bytes.

    Returns (payload, (w, h), quality) or (None, ..., ...) if even the
    smallest setting overflows -- which happens when the budget is below the
    codec's own container overhead.
    """
    best = None
    for scale in SCALES:
        for q in QUALITIES:
            data, dims = encode_at(img, codec, scale, q)
            if data is not None and len(data) <= target_bytes:
                # First hit at this scale is the best quality that fits here,
                # and scales are ordered large -> small, so take it and stop.
                return data, dims, q
            best = (data, dims, q)
    return None, best[1] if best else (0, 0), best[2] if best else 0


def main():
    ap = argparse.ArgumentParser(
        description="Encode images as transmittable payloads (the pixel baseline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--meta", required=True,
                    help="GBSED scene_data directory -- defines frame order and, "
                         "in matched mode, the per-frame byte budget")
    ap.add_argument("--images", required=True,
                    help="directory holding the source images")
    ap.add_argument("--out", required=True, help="output directory for .bin files")
    ap.add_argument("--mode", choices=["full", "matched"], default="matched")
    ap.add_argument("--codec", choices=["jpeg", "webp"], default="jpeg",
                    help="codec for matched mode; webp is markedly better at "
                         "very low bitrates, so it is the fairer baseline")
    ap.add_argument("--budget-scale", type=float, default=1.0,
                    help="multiply the GBSED byte budget by this, to explore "
                         "how much MORE bandwidth pixels would need")
    ap.add_argument("--stage", default=None, metavar="SCENARIO_DIR",
                    help="also copy the .bin files into SCENARIO_DIR/scene_data")
    args = ap.parse_args()

    meta_dir = Path(args.meta).expanduser().resolve()
    img_dir = Path(args.images).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    metas = sorted(glob.glob(str(meta_dir / "*.meta.json")))
    if not metas:
        raise SystemExit("no *.meta.json in %s" % meta_dir)

    print("mode             : %s%s" % (args.mode,
          "" if args.mode == "full" else "  (codec=%s, budget x%.2f)"
          % (args.codec, args.budget_scale)))
    print("frame order from : %s (%d frames)" % (meta_dir, len(metas)))
    print("images           : %s" % img_dir)
    print("output           : %s" % outdir)
    print()

    hdr = "%-5s %-16s %10s %10s %12s %8s  %s" % (
        "idx", "image", "gbsed_B", "image_B", "resolution", "quality", "note")
    print(hdr)
    print("-" * len(hdr))

    entries = []
    for mp in metas:
        gm = json.load(open(mp))
        name = os.path.basename(gm["source_image"])
        src = img_dir / name
        if not src.is_file():
            raise SystemExit("source image not found: %s\n"
                             "(meta refers to %s)" % (src, gm["source_image"]))

        t0 = time.time()
        budget = int(round(gm["n_bytes"] * args.budget_scale))
        note = ""

        if args.mode == "full":
            payload = src.read_bytes()
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            dims = (img.shape[1], img.shape[0]) if img is not None else (0, 0)
            quality = None
        else:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit("could not read %s" % src)
            img = cv2.resize(img, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
            payload, dims, quality = fit_to_budget(img, args.codec, budget)
            if payload is None:
                # Below the codec's container floor. Emit the smallest we can
                # and record it -- this is itself a result worth reporting.
                payload, dims, quality = encode_at(img, args.codec,
                                                   SCALES[-1], QUALITIES[-1]) + (QUALITIES[-1],)
                payload = payload if isinstance(payload, bytes) else b""
                note = "OVER BUDGET (codec floor)"

        stem = gm["stem"]
        (outdir / (stem + ".bin")).write_bytes(payload)

        meta = {
            "frame": gm["frame"], "stem": stem, "bin": stem + ".bin",
            "source_image": str(src),
            "mode": args.mode, "codec": None if args.mode == "full" else args.codec,
            "n_bytes": len(payload),
            "gbsed_n_bytes": gm["n_bytes"],
            "budget_bytes": budget,
            "resolution": list(dims), "quality": quality,
            "sha256": __import__("hashlib").sha256(payload).hexdigest(),
            # carried through so the decoder can score against the same truth
            "graph": gm["graph"],
            "encode_seconds": round(time.time() - t0, 3),
            "note": note,
        }
        (outdir / (stem + ".meta.json")).write_text(json.dumps(meta, indent=2))

        print("%-5d %-16s %10d %10d %12s %8s  %s"
              % (gm["frame"], name[:16], gm["n_bytes"], len(payload),
                 "%dx%d" % (dims[0], dims[1]),
                 "-" if quality is None else quality, note))
        entries.append(meta)

    file_path_param = ";".join("scene_data/" + e["bin"] for e in entries)
    total = sum(e["n_bytes"] for e in entries)
    gtotal = sum(e["gbsed_n_bytes"] for e in entries)
    chunk = 1000
    n_chunks = sum(-(-e["n_bytes"] // chunk) for e in entries)

    manifest = {
        "experiment": "image_baseline", "mode": args.mode,
        "codec": None if args.mode == "full" else args.codec,
        "budget_scale": args.budget_scale,
        "n_frames": len(entries), "total_bytes": total,
        "gbsed_total_bytes": gtotal,
        "expansion_vs_gbsed": round(total / gtotal, 2) if gtotal else None,
        "n_chunks": n_chunks,
        "frames": [{"frame": e["frame"], "bin": e["bin"], "n_bytes": e["n_bytes"],
                    "resolution": e["resolution"], "quality": e["quality"],
                    "sha256": e["sha256"]} for e in entries],
        "omnetpp_filePath": file_path_param,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.stage:
        staged = Path(args.stage).expanduser().resolve() / "scene_data"
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir(parents=True, exist_ok=True)
        for e in entries:
            shutil.copy2(outdir / e["bin"], staged / e["bin"])
        print("\nstaged to        : %s" % staged)

    print()
    print("total payload    : %s B  (GBSED: %s B, expansion x%s)"
          % (f"{total:,}", f"{gtotal:,}", manifest["expansion_vs_gbsed"]))
    print("chunks @%dB      : %d  (GBSED: %d)"
          % (chunk, n_chunks, sum(-(-e["gbsed_n_bytes"] // chunk) for e in entries)))
    print("manifest         : %s" % (outdir / "manifest.json"))
    if args.mode == "full":
        secs = 2.5 * n_chunks
        print()
        print("At one chunk every 2.5 s this queue needs %.0f s of simulated time"
              % secs)
        print("(%.1f minutes). The sender vehicle exists for ~125 s." % (secs / 60))


if __name__ == "__main__":
    main()
