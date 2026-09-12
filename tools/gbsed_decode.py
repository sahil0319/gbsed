#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GBSED decoder -- the .bin files a Veins receiver wrote back to scene graphs.

    .bin -> format_loading() -> sem_decompression() -> decode() -> SceneGraph'

For every frame the encoder produced, this looks for the receiver's copy,
rebuilds the scene graph, and scores it against the `meta.json` ground truth.
Frames the receiver never completed are reported as LOST rather than silently
missing -- that is the interesting outcome in a VANET, not an error.

Example
-------
    python tools/gbsed_decode.py \
        --received /path/to/gbsed_veins/src/veins/.../GBSEDApp/received \
        --meta scene_data --out decoded --visualize

Set GBSED_VEINS_APP to the scenario directory once and --received can be
omitted; see RUNNING.md in the Veins repository.
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gbsed_semantic as gs


def read_chunk_log(path):
    """Fold a tx_log.csv / rx_log.csv into one record per file: when its last
    chunk was seen, how far apart the vehicles were, how many chunks."""
    if not path.is_file():
        return {}
    per_file = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("fileName", "")
            if not name:
                continue
            rec = per_file.setdefault(
                name, {"chunks": 0, "first_time": None, "last_time": None,
                       "max_distance": None, "total_chunks": None})
            rec["chunks"] += 1
            try:
                t = float(row["simTime"])
                rec["first_time"] = t if rec["first_time"] is None else min(rec["first_time"], t)
                rec["last_time"] = t if rec["last_time"] is None else max(rec["last_time"], t)
            except (KeyError, ValueError):
                pass
            try:
                d = float(row["distance"])
                rec["max_distance"] = d if rec["max_distance"] is None \
                    else max(rec["max_distance"], d)
            except (KeyError, ValueError):
                pass                                  # tx_log has no distance column filled
            try:
                rec["total_chunks"] = int(row["totalChunks"])
            except (KeyError, ValueError):
                pass
    return per_file


def find_received(received_dir, meta):
    """The receiver writes `received_<basename>`; fall back to the bare name
    so a decode straight out of the encoder's own folder also works."""
    for name in ("received_" + meta["bin"], meta["bin"]):
        p = received_dir / name
        if p.is_file():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Decode received GBSED payloads back into scene graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _app = os.environ.get("GBSED_VEINS_APP")
    ap.add_argument("--received", default=(_app + "/received") if _app else None,
                    required=not _app,
                    help="directory holding the receiver's .bin files "
                         "(defaults to $GBSED_VEINS_APP/received)")
    ap.add_argument("--meta", default="scene_data",
                    help="directory holding the encoder's *.meta.json")
    ap.add_argument("--out", default="decoded",
                    help="directory for rendered graphs and fidelity.csv")
    ap.add_argument("--config", default=None,
                    help="path to pipeline_extraction.yaml (must match the encoder's)")
    ap.add_argument("--visualize", action="store_true",
                    help="render each reconstructed graph to <out>/png/")
    ap.add_argument("--verbose", action="store_true",
                    help="print the full node/edge diff for every frame")
    args = ap.parse_args()

    received_dir = Path(args.received).expanduser().resolve()
    meta_dir = Path(args.meta).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    pngdir = outdir / "png"
    if args.visualize:
        pngdir.mkdir(exist_ok=True)

    metas = sorted(meta_dir.glob("*.meta.json"))
    if not metas:
        raise SystemExit("No *.meta.json found in %s -- run gbsed_encode.py first."
                         % meta_dir)

    cfg = gs.load_config(args.config)
    ae, _ = gs.make_autoencoder(cfg)
    fingerprint = gs.codebook_fingerprint(cfg)

    print("config           : %s" % cfg.yaml_path)
    print("codebook         : %s" % fingerprint)
    print("received from    : %s" % received_dir)
    print("ground truth     : %s (%d frame(s))" % (meta_dir, len(metas)))

    tx_log = read_chunk_log(received_dir / "tx_log.csv")
    rx_log = read_chunk_log(received_dir / "rx_log.csv")
    if tx_log or rx_log:
        print("chunk logs       : %d file(s) sent, %d file(s) heard from"
              % (len(tx_log), len(rx_log)))
    print()

    rows = []
    hdr = "%-5s %-10s %7s %8s %11s %11s %8s  %s" % (
        "idx", "status", "bytes", "dist_m", "nodes o/r", "edges o/r", "edge F1", "note")
    print(hdr)
    print("-" * len(hdr))

    for mp in metas:
        meta = json.loads(mp.read_text())
        idx = meta["frame"]
        row = {
            "frame": idx, "source_image": meta["source_image"],
            "sent_bytes": meta["n_bytes"], "received_bytes": 0,
            "status": "LOST", "bit_exact": False,
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
            "format": meta.get("format", "v1"),
            "blocks_missing": "", "relations_recovered": "",
            "relations_sent": len(meta.get("active_relation_indexes", []) or []),
            "tx_time": "", "rx_time": "", "distance_m": "",
            "chunks_sent": "", "chunks_heard": 0,
            "note": "",
        }

        tx = tx_log.get(meta["bin"])
        if tx:
            row["tx_time"] = round(tx["first_time"], 2) if tx["first_time"] is not None else ""
            row["chunks_sent"] = tx["chunks"]
        rx = rx_log.get(meta["bin"])
        if rx:
            row["rx_time"] = round(rx["last_time"], 2) if rx["last_time"] is not None else ""
            row["distance_m"] = round(rx["max_distance"], 1) \
                if rx["max_distance"] is not None else ""
            row["chunks_heard"] = rx["chunks"]

        if meta.get("codebook") != fingerprint:
            row["note"] = "codebook mismatch (%s vs %s)" % (
                meta.get("codebook"), fingerprint)

        path = find_received(received_dir, meta)
        if path is None:
            if not row["note"]:
                heard, total = row["chunks_heard"], (rx or {}).get("total_chunks")
                if heard and total:
                    # Partial reception is the interesting case: the frame was
                    # in range for some of its chunks but not all of them.
                    row["note"] = "only %d of %d chunks arrived" % (heard, total)
                elif row["chunks_sent"]:
                    row["note"] = "sent at t=%s, nothing heard" % row["tx_time"]
                else:
                    row["note"] = "never transmitted"
            rows.append(row)
            print("%-5d %-10s %7s %8s %11s %11s %8s  %s"
                  % (idx, "LOST", "-", row["distance_m"] or "-",
                     "-", "-", "-", row["note"]))
            continue

        raw = path.read_bytes()
        row["received_bytes"] = len(raw)
        row["bit_exact"] = (gs.sha256_of(raw) == meta["sha256"])

        try:
            # meta is authoritative about the format; sniffing is the fallback
            # for payloads decoded without their metadata.
            rec_sg, _detail = gs.decode_payload(
                ae, raw, meta.get("chunk_size"), meta.get("format"))
        except Exception as e:
            row["status"] = "CORRUPT"
            row["note"] = "%s: %s" % (type(e).__name__, e)
            rows.append(row)
            print("%-5d %-10s %7d %8s %11s %11s %8s  %s"
                  % (idx, "CORRUPT", len(raw), row["distance_m"] or "-",
                     "-", "-", "-", row["note"][:38]))
            continue

        ok, lines, stats = gs.compare_summary(meta["graph"], rec_sg)
        em = gs.edge_metrics(meta["graph"]["edges"], gs.edge_set(rec_sg))

        n_rec, n_orig = stats["n_edges_rec"], stats["n_edges_orig"]
        f1 = em["edge_f1"]

        row.update({
            "status": "EXACT" if ok else "DEGRADED",
            "nodes_rec": stats["n_nodes_rec"], "edges_rec": n_rec,
            "edges_common": stats["n_edges_common"],
            "nodes_match": stats["nodes_match"],
        })
        row.update(em)

        # v2 only: which blocks failed the magic check, and therefore which
        # relation slices never arrived. Without this a partially delivered
        # frame is indistinguishable from a genuinely sparse scene.
        row["blocks_missing"] = len(_detail.get("blocks_missing", []) or [])
        row["relations_recovered"] = len(_detail.get("relations_recovered", []) or [])
        if row["blocks_missing"]:
            row["note"] = "partial: %d block(s) lost, %d/%d relations recovered" % (
                row["blocks_missing"], row["relations_recovered"],
                row["relations_sent"])

        if not row["bit_exact"] and not row["note"]:
            row["note"] = "payload differs from sent bytes"

        if args.visualize:
            gs.maybe_visualize(rec_sg, pngdir / (meta["stem"] + "_reconstructed.png"))

        print("%-5d %-10s %7d %8s %11s %11s %8.3f  %s"
              % (idx, row["status"], len(raw), row["distance_m"] or "-",
                 "%d/%d" % (stats["n_nodes_orig"], stats["n_nodes_rec"]),
                 "%d/%d" % (n_orig, n_rec), f1, row["note"]))
        if args.verbose or not ok:
            for line in lines:
                print("      " + line)
        rows.append(row)

    # ---- summary --------------------------------------------------------
    csv_path = outdir / "fidelity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    delivered = [r for r in rows if r["status"] != "LOST"]
    exact = [r for r in rows if r["status"] == "EXACT"]
    degraded = [r for r in rows if r["status"] == "DEGRADED"]
    corrupt = [r for r in rows if r["status"] == "CORRUPT"]
    bit_exact = [r for r in delivered if r["bit_exact"]]

    print()
    print("=" * len(hdr))
    print("frames           : %d" % n)
    print("delivered        : %d (%.1f%%)" % (len(delivered), 100.0 * len(delivered) / n))
    print("  bit-exact      : %d" % len(bit_exact))
    print("semantic fidelity: %d/%d exact (%.1f%%)"
          % (len(exact), n, 100.0 * len(exact) / n))
    if degraded:
        print("  degraded       : %d" % len(degraded))
    if corrupt:
        print("  undecodable    : %d" % len(corrupt))
    if delivered:
        print("mean edge F1     : %.3f | delivered, %.3f | all frames"
              % (sum(r["edge_f1"] for r in delivered) / len(delivered),
                 sum(r["edge_f1"] for r in rows) / n))
        print("mean actor F1    : %.3f | delivered  (skeleton excluded)"
              % (sum(r["actor_edge_f1"] for r in delivered) / len(delivered)))
    ro = sum(r["risky_orig"] for r in delivered)
    rp = sum(r["risky_preserved"] for r in delivered)
    print("safety relations : %d of %d preserved on delivered frames%s"
          % (rp, ro, "" if not ro else "  (%.1f%%)" % (100.0 * rp / ro)))
    print("fidelity.csv     : %s" % csv_path)
    if args.visualize:
        print("renders          : %s" % pngdir)


if __name__ == "__main__":
    main()
