#!/usr/bin/env bash
#
# run_experiment_matrix.sh -- sweep multiple encoded sequences across all
# channel-condition configs defined in omnetpp.ini, decode + classify each
# result, and log everything to one results CSV.
#
# Prerequisites (must already exist before running this):
#   - gbsed/scene_data_seq1/  (from `gbsed_encode.py`, sequence 1 already encoded)
#   - gbsed/scene_data_seq2/  (sequence 2 already encoded)
#   - gbsed/checkpoints/1043_task_oriented_model.pt
#   - The venv activated (source /home/opp_env/.venv/bin/activate)
#   - omnetpp.ini already has the [Config ...] blocks (Baseline, Noise_Low,
#     Noise_Medium, Noise_High, CAV_Good, CAV_Moderate, CAV_Bad, CAV_Extreme,
#     and CAV_Failure if you've added it -- edit CONFIGS below to match
#     whatever's actually in the ini)
#
# What it does, per sequence, per config:
#   1. Stages that sequence's .bin files into the Veins scenario directory
#      and rewrites the ini's filePath to match (only needs doing once per
#      sequence, not per config, since configs only add settings on top of
#      General via `extends = General`)
#   2. Clears + recreates `received/` (required -- GBSEDApp doesn't create
#      this directory itself, see project notes)
#   3. Runs the simulation with that config
#   4. Archives the received files under a per-(sequence,config) folder so
#      the next run doesn't overwrite them
#   5. Decodes (fidelity/edge-F1 metrics) and classifies (risk prediction)
#   6. Appends one row to results_matrix.csv
#
# Usage:
#   ./run_experiment_matrix.sh
#
# Edit the paths and CONFIGS list below to match your setup first.

set -uo pipefail
# NOT `set -e`: a config in which nothing is delivered makes the risk-assess
# step exit non-zero, and with -e that aborted the whole sweep -- which is why
# CAV_Extreme ran but never reached results_matrix.csv. Zero delivery is the
# most interesting data point, not a failure.

# --------------------------------------------------------------------------
# Configuration -- edit these to match your environment
# --------------------------------------------------------------------------
GBSED_REPO="/home/opp_env/default_workspace/gbsed"
VEINS_REPO="/home/opp_env/default_workspace/veins"
GBSED_VEINS_APP="$VEINS_REPO/src/veins/modules/application/gbsed/GBSEDApp"
CHECKPOINT="$GBSED_REPO/checkpoints/1043_task_oriented_model.pt"
LEARNING_CFG="$GBSED_REPO/Config/pipeline_learning.yaml"

# Sequence name -> scene_data directory (relative to $GBSED_REPO)
declare -A SEQUENCES=(
    [seq1]="scene_data_seq1"
#    [seq2]="scene_data_seq2"
)

# Channel-condition configs to sweep. Must match [Config <name>] blocks in
# omnetpp.ini exactly. Add/remove entries here if your ini differs (e.g.
# once CAV_Failure is merged in).
# CAV_Good / CAV_Moderate / CAV_Bad removed: they set node[1].veinsmobility.x,
# which TraCI overwrites, so they were exact duplicates of Baseline /
# Noise_Medium / Noise_High. Five distinct noise floors remain.
CONFIGS=(Baseline Noise_Low Noise_Medium Noise_High CAV_Extreme)

RESULTS_DIR="$GBSED_REPO/experiment_results"
RESULTS_CSV="$RESULTS_DIR/results_matrix.csv"

# --------------------------------------------------------------------------
mkdir -p "$RESULTS_DIR"
export GBSED_VEINS_APP

if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "sequence,config,frames_total,frames_delivered,frames_exact,delivery_rate,graph_exact_rate,bit_exact_rate,mean_edge_f1_delivered,mean_edge_f1_all,mean_actor_edge_f1_delivered,risky_total,risky_preserved,risky_recall,prediction,prob_class0,prob_class1,ground_truth,match,n_frames_classified" > "$RESULTS_CSV"
fi

stage_sequence () {
    local scene_data_dir="$1"

    echo ">> Staging $scene_data_dir into $GBSED_VEINS_APP/scene_data ..."
    rm -rf "$GBSED_VEINS_APP/scene_data"
    mkdir -p "$GBSED_VEINS_APP/scene_data"
    cp "$GBSED_REPO/$scene_data_dir"/*.bin "$GBSED_VEINS_APP/scene_data/"

    echo ">> Rewriting omnetpp.ini filePath for $scene_data_dir ..."
    python3 - "$scene_data_dir" <<'PY'
import json, os, re, sys, pathlib
scene_data_dir = sys.argv[1]
gbsed_repo = os.environ["GBSED_REPO_PY"]
app = pathlib.Path(os.environ["GBSED_VEINS_APP"])
ini_path = app / "omnetpp.ini"
manifest = json.load(open(f"{gbsed_repo}/{scene_data_dir}/manifest.json"))
fp = manifest["omnetpp_filePath"]
text = ini_path.read_text()
text = re.sub(r'^\*\.node\[0\]\.appl\.filePath = .*$',
              '*.node[0].appl.filePath = "%s"' % fp, text, flags=re.M)
ini_path.write_text(text)
PY
}

run_one_config () {
    local seq_name="$1" scene_data_dir="$2" config="$3"

    echo ""
    echo "=========================================================="
    echo ">> $seq_name / $config"
    echo "=========================================================="

    rm -rf "$GBSED_VEINS_APP/received"
    mkdir -p "$GBSED_VEINS_APP/received"

    ( cd "$VEINS_REPO" && ./run_gbsed.sh -c "$config" -u Cmdenv )

    local archived_received="$RESULTS_DIR/received_${seq_name}_${config}"
    rm -rf "$archived_received"
    cp -r "$GBSED_VEINS_APP/received" "$archived_received"

    local decoded_out="$RESULTS_DIR/decoded_${seq_name}_${config}"
    ( cd "$GBSED_REPO" && python tools/gbsed_decode.py \
        --received "$archived_received" \
        --meta "$scene_data_dir" \
        --out "$decoded_out" )

    rm -f "$GBSED_REPO/risk_prediction.json"
    ( cd "$GBSED_REPO" && python tools/run_risk_assess_compat.py \
        --received "$archived_received" \
        --meta "$scene_data_dir" \
        --checkpoint "$CHECKPOINT" \
        --config "$LEARNING_CFG" \
        --device cpu ) || echo ">> risk assessment produced no prediction "\
        "(expected when nothing was delivered); logging the row anyway"

    local pred_json="$RESULTS_DIR/risk_prediction_${seq_name}_${config}.json"
    if [[ -f "$GBSED_REPO/risk_prediction.json" ]]; then
        mv "$GBSED_REPO/risk_prediction.json" "$pred_json"
    else
        rm -f "$pred_json"
    fi

    local gt_json="$RESULTS_DIR/ground_truth_${seq_name}.json"
    if [[ ! -f "$gt_json" ]]; then
        ( cd "$GBSED_REPO" && python tools/generate_ground_truth.py \
            --meta "$scene_data_dir" --out "$gt_json" )
    fi

    # Aggregate this run's metrics into one CSV row.
    python3 - "$seq_name" "$config" "$decoded_out/fidelity.csv" "$pred_json" "$gt_json" "$RESULTS_CSV" <<'PY'
import csv, json, os, sys

seq_name, config, fidelity_csv, pred_json, gt_json, results_csv = sys.argv[1:7]

rows = list(csv.DictReader(open(fidelity_csv)))
n = len(rows)

# A frame is "delivered" if it arrived at all; "exact" if the reconstruction
# matches. For GBSED these coincide (802.11p hands up an intact frame or none),
# but keeping them separate is what lets the pixel baseline be compared on the
# same axes.
arrived = [r for r in rows if r["status"] != "LOST"]
exact = [r for r in rows if r["status"] == "EXACT"]
bit_exact = sum(1 for r in rows if r["bit_exact"] == "True")

# The previous filter was `if r["edge_f1"]`, meant to skip LOST frames -- but
# their edge_f1 is the STRING "0.0", which is truthy, so they were counted as
# zeros. Since every delivered frame scores exactly 1.0, the mean collapsed to
# the delivery rate and the column carried no information of its own.
f1_deliv = [float(r["edge_f1"]) for r in arrived]
f1_all = [float(r["edge_f1"]) for r in rows]
mean_f1_deliv = sum(f1_deliv) / len(f1_deliv) if f1_deliv else 0.0
mean_f1_all = sum(f1_all) / len(f1_all) if f1_all else 0.0

# Skeleton-free view: edges involving a detected actor. Plain edge F1 has a
# floor near 0.38 that comes free from the road/ego/lane structure.
def col(r, name, default=0.0):
    v = r.get(name, "")
    return float(v) if v not in ("", None) else default

actor_f1 = [col(r, "actor_edge_f1") for r in arrived]
mean_actor_f1 = sum(actor_f1) / len(actor_f1) if actor_f1 else 0.0

# The task-level metric: did the relations a braking decision reads survive?
risky_total = sum(int(col(r, "risky_orig")) for r in arrived)
risky_kept = sum(int(col(r, "risky_preserved")) for r in arrived)

prediction, probs = None, ["", ""]
if os.path.isfile(pred_json):
    pred = json.load(open(pred_json))
    prediction = pred.get("prediction")
    if isinstance(prediction, list):
        prediction = prediction[0] if prediction else None
    probs = pred.get("class_probabilities", ["", ""])
    if isinstance(probs, list) and len(probs) == 2 and isinstance(probs[0], list):
        probs = probs[0]

gt = json.load(open(gt_json))
gt_label = gt["sequence_label"]

match = "" if prediction is None else int(prediction) == int(gt_label)
# n_frames_classified is logged alongside the prediction: as delivery falls the
# classifier sees both fewer AND different frames, so a flipped prediction
# cannot be attributed to degraded semantics without it.

with open(results_csv, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        seq_name, config, n, len(arrived), len(exact),
        round(len(arrived) / n, 4) if n else "",
        round(len(exact) / n, 4) if n else "",
        round(bit_exact / n, 4) if n else "",
        round(mean_f1_deliv, 4), round(mean_f1_all, 4),
        round(mean_actor_f1, 4),
        risky_total, risky_kept,
        round(risky_kept / risky_total, 4) if risky_total else "",
        prediction,
        probs[0] if len(probs) > 0 else "",
        probs[1] if len(probs) > 1 else "",
        gt_label, match, len(arrived),
    ])

print(f"Logged: {seq_name}/{config}  delivered={len(arrived)}/{n}  "
      f"exact={len(exact)}  actor_f1={mean_actor_f1:.3f}  "
      f"safety={risky_kept}/{risky_total}  prediction={prediction}  "
      f"gt={gt_label}  match={match}")
PY
}

export GBSED_REPO_PY="$GBSED_REPO"

for seq_name in "${!SEQUENCES[@]}"; do
    scene_data_dir="${SEQUENCES[$seq_name]}"
    stage_sequence "$scene_data_dir"
    for config in "${CONFIGS[@]}"; do
        run_one_config "$seq_name" "$scene_data_dir" "$config"
    done
done

echo ""
echo "=========================================================="
echo "Done. Results written to: $RESULTS_CSV"
echo "=========================================================="
column -s, -t "$RESULTS_CSV" 2>/dev/null || cat "$RESULTS_CSV"
