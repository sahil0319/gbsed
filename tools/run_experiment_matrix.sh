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

set -euo pipefail

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
CONFIGS=(Baseline Noise_Low Noise_Medium Noise_High CAV_Good CAV_Moderate CAV_Bad CAV_Extreme)

RESULTS_DIR="$GBSED_REPO/experiment_results"
RESULTS_CSV="$RESULTS_DIR/results_matrix.csv"

# --------------------------------------------------------------------------
mkdir -p "$RESULTS_DIR"
export GBSED_VEINS_APP

if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "sequence,config,frames_total,frames_delivered,delivery_rate,bit_exact_rate,mean_edge_f1,prediction,prob_class0,prob_class1,ground_truth,match" > "$RESULTS_CSV"
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

    ( cd "$GBSED_REPO" && python tools/run_risk_assess_compat.py \
        --received "$archived_received" \
        --meta "$scene_data_dir" \
        --checkpoint "$CHECKPOINT" \
        --config "$LEARNING_CFG" \
        --device cpu )

    local pred_json="$RESULTS_DIR/risk_prediction_${seq_name}_${config}.json"
    mv "$GBSED_REPO/risk_prediction.json" "$pred_json"

    local gt_json="$RESULTS_DIR/ground_truth_${seq_name}.json"
    if [[ ! -f "$gt_json" ]]; then
        ( cd "$GBSED_REPO" && python tools/generate_ground_truth.py \
            --meta "$scene_data_dir" --out "$gt_json" )
    fi

    # Aggregate this run's metrics into one CSV row.
    python3 - "$seq_name" "$config" "$decoded_out/fidelity.csv" "$pred_json" "$gt_json" "$RESULTS_CSV" <<'PY'
import csv, json, sys

seq_name, config, fidelity_csv, pred_json, gt_json, results_csv = sys.argv[1:7]

rows = list(csv.DictReader(open(fidelity_csv)))
n = len(rows)
delivered = sum(1 for r in rows if r["status"] == "EXACT")
bit_exact = sum(1 for r in rows if r["bit_exact"] == "True")
f1s = [float(r["edge_f1"]) for r in rows if r["edge_f1"]]
mean_f1 = sum(f1s) / len(f1s) if f1s else ""

pred = json.load(open(pred_json))
prediction = pred["prediction"]
if isinstance(prediction, list):
    prediction = prediction[0] if prediction else None
probs = pred.get("class_probabilities", ["", ""])
if isinstance(probs, list) and len(probs) == 2 and isinstance(probs[0], list):
    probs = probs[0]  # in case it's nested [[p0, p1]]

gt = json.load(open(gt_json))
gt_label = gt["sequence_label"]

match = "" if prediction is None else int(prediction) == int(gt_label)

with open(results_csv, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        seq_name, config, n, delivered,
        round(delivered / n, 4) if n else "",
        round(bit_exact / n, 4) if n else "",
        round(mean_f1, 4) if mean_f1 != "" else "",
        prediction,
        probs[0] if len(probs) > 0 else "",
        probs[1] if len(probs) > 1 else "",
        gt_label, match,
    ])

print(f"Logged: {seq_name}/{config}  delivered={delivered}/{n}  "
      f"mean_f1={mean_f1}  prediction={prediction}  gt={gt_label}  match={match}")
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
