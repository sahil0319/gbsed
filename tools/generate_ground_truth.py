#!/usr/bin/env python3
"""
generate_ground_truth.py -- derive a sequence-level "risky / not risky"
ground-truth label from the scene graph relations the encoder already
extracted (scene_data/*.meta.json), as a proxy for human-annotated labels
from the restricted 1043 dataset.

Rule: if ANY frame in the sequence has a proximity relation at or above a
chosen severity level directly involving 'ego car' (as source OR target),
the whole sequence is labeled risky (1). Otherwise not risky (0).

Severity order, closest first (from Config/pipeline_extraction.yaml's
PROXIMITY_THRESHOLDS, in decreasing order of closeness):
    near_coll (4 ft) > super_near (7 ft) > very_near (10 ft) > near (16 ft) > visible (25 ft)

Default risky set is {near_coll, super_near} -- tune with --risky-relations
if you want a stricter or looser definition (e.g. include very_near).

This is a HEURISTIC proxy label, not a substitute for real human-annotated
ground truth. It only reflects geometric proximity as computed by the
extractor's distance thresholds, not actual collision risk (speed, closing
rate, driver behavior, etc. are not modeled). Treat comparisons against it
as a sanity check, not a rigorous accuracy evaluation.

Usage:
    python tools/generate_ground_truth.py --meta scene_data
    python tools/generate_ground_truth.py --meta scene_data --risky-relations near_coll super_near very_near
    python tools/generate_ground_truth.py --meta scene_data --compare risk_prediction.json
"""
import argparse
import json
from pathlib import Path

DEFAULT_RISKY_RELATIONS = {"near_coll", "super_near"}
EGO_NODE_NAME = "ego car"  # matches the "name" field roadscene2vec assigns the ego node


def label_frame(meta, risky_relations):
    """Return (is_risky, matching_edges) for one frame's meta.json."""
    edges = meta["graph"]["edges"]
    hits = []
    for src, rel, dst in edges:
        if rel in risky_relations and (src == EGO_NODE_NAME or dst == EGO_NODE_NAME):
            other = dst if src == EGO_NODE_NAME else src
            hits.append({"relation": rel, "other_actor": other})
    return (len(hits) > 0, hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="path to scene_data directory")
    ap.add_argument("--risky-relations", nargs="+", default=sorted(DEFAULT_RISKY_RELATIONS),
                     help="relation names that count as 'risky' if they involve ego car "
                          "(default: %(default)s)")
    ap.add_argument("--compare", default=None,
                     help="optional path to risk_prediction.json from risk_assess.py, "
                          "to print a match/mismatch report")
    ap.add_argument("--out", default=None,
                     help="where to write the ground truth json "
                          "(default: <meta>/../ground_truth.json)")
    args = ap.parse_args()

    risky_relations = set(args.risky_relations)
    meta_dir = Path(args.meta)
    meta_files = sorted(meta_dir.glob("*.meta.json"),
                         key=lambda p: json.loads(p.read_text())["frame"])
    if not meta_files:
        raise SystemExit(f"No *.meta.json files found in {meta_dir}")

    print(f"Risky relation set: {sorted(risky_relations)}\n")
    print(f"{'frame':<6} {'risky?':<8} details")
    print("-" * 70)

    frame_labels = []
    any_risky = False
    for mf in meta_files:
        meta = json.loads(mf.read_text())
        is_risky, hits = label_frame(meta, risky_relations)
        any_risky = any_risky or is_risky
        detail = ", ".join(f"{h['relation']}({h['other_actor']})" for h in hits) or "-"
        print(f"{meta['stem']:<6} {'YES' if is_risky else 'no':<8} {detail}")
        frame_labels.append({
            "frame": meta["frame"], "stem": meta["stem"],
            "risky": is_risky, "matching_edges": hits,
        })

    sequence_label = 1 if any_risky else 0
    print("\n" + "=" * 70)
    print(f"SEQUENCE-LEVEL GROUND TRUTH: {sequence_label} "
          f"({'RISKY' if sequence_label else 'NOT RISKY'})")
    print("=" * 70)

    result = {
        "risky_relations_used": sorted(risky_relations),
        "sequence_label": sequence_label,
        "sequence_label_meaning": "1=risky, 0=not risky",
        "per_frame": frame_labels,
        "note": ("Heuristic proxy ground truth derived from geometric proximity "
                 "relations, not human-annotated. See script docstring."),
    }

    out_path = Path(args.out) if args.out else meta_dir.parent / "ground_truth.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote ground truth to {out_path}")

    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.is_file():
            print(f"\n(--compare path not found: {compare_path}, skipping comparison)")
            return
        pred = json.loads(compare_path.read_text())
        predicted = pred.get("prediction")
        # prediction may be a list (e.g. [1]) or a bare int depending on how
        # risk_assess.py serialized it -- normalize to a single int
        if isinstance(predicted, list):
            predicted = predicted[0] if predicted else None

        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"Ground truth : {sequence_label} ({'RISKY' if sequence_label else 'NOT RISKY'})")
        print(f"Model predicted: {predicted}")
        if predicted is None:
            print("Could not read a prediction value from the compare file.")
        elif int(predicted) == sequence_label:
            print("Result: MATCH")
        else:
            print("Result: MISMATCH")
        print("\nNote: this is a single-sequence comparison (n=1). It tells you "
              "whether the model agreed with this one heuristic label -- it is "
              "not a statistically meaningful accuracy/precision/recall estimate. "
              "That requires many independently labeled sequences.")


if __name__ == "__main__":
    main()
