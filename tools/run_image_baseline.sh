#!/usr/bin/env bash
#
# run_image_baseline.sh -- sweep the pixel baseline across the same channel
# configs the GBSED experiment matrix used, so the two are directly comparable.
#
# Unlike run_experiment_matrix.sh this does NOT use `set -e` around the per-run
# body: a config in which nothing is delivered is a legitimate result, not an
# error, and aborting on it silently drops the most interesting data point.
#
#   GBSED_REPO / VEINS_REPO override the defaults below.
#
set -uo pipefail

GBSED_REPO="${GBSED_REPO:-$HOME/Desktop/PythonEnvs/gbsed}"
VEINS_REPO="${VEINS_REPO:-$HOME/Documents/gbsed_veins}"
APP="$VEINS_REPO/src/veins/modules/application/gbsed/GBSEDApp"
PY="${PY:-$HOME/miniconda3/envs/av/bin/python}"
RESULTS="$GBSED_REPO/experiment_results_image"

# Only the DISTINCT channel conditions. CAV_Good/Moderate/Bad duplicate
# Baseline/Noise_Medium/Noise_High exactly -- see the analysis writeup.
CONFIGS=(Baseline Noise_Low Noise_Medium Noise_High CAV_Extreme)

PAYLOAD_SETS=("$@")
if [ ${#PAYLOAD_SETS[@]} -eq 0 ]; then
    PAYLOAD_SETS=(image_data_seq1_webp image_data_seq1_jpeg)
fi

mkdir -p "$RESULTS"
CSV="$RESULTS/image_results_matrix.csv"
[ -f "$CSV" ] || echo "payload_set,config,frames_total,frames_delivered,delivery_rate,graph_exact,mean_edge_f1_delivered,mean_edge_f1_all,total_detections,mean_psnr_db" > "$CSV"

for SET in "${PAYLOAD_SETS[@]}"; do
    echo ""
    echo ">> staging $SET"
    rm -rf "$APP/scene_data"; mkdir -p "$APP/scene_data"
    cp "$GBSED_REPO/$SET"/*.bin "$APP/scene_data/"
    "$PY" - "$GBSED_REPO/$SET" "$APP/omnetpp.ini" <<'PY'
import json, re, sys, pathlib
setdir, ini = sys.argv[1], pathlib.Path(sys.argv[2])
fp = json.load(open(f"{setdir}/manifest.json"))["omnetpp_filePath"]
ini.write_text(re.sub(r'^\*\.node\[0\]\.appl\.filePath = .*$',
                      '*.node[0].appl.filePath = "%s"' % fp,
                      ini.read_text(), flags=re.M))
print("   filePath ->", fp.count(";") + 1, "frames")
PY

    for CFG in "${CONFIGS[@]}"; do
        echo ""
        echo "=== $SET / $CFG ==="
        rm -rf "$APP/received"; mkdir -p "$APP/received"
        ( cd "$VEINS_REPO" && ./run_gbsed.sh -c "$CFG" -u Cmdenv ) >/dev/null 2>&1

        ARCH="$RESULTS/received_${SET}_${CFG}"
        rm -rf "$ARCH"; cp -r "$APP/received" "$ARCH"

        DEC="$RESULTS/decoded_${SET}_${CFG}"
        "$PY" "$GBSED_REPO/tools/image_decode.py" \
            --received "$ARCH" --meta "$GBSED_REPO/$SET" --out "$DEC" 2>&1 \
            | grep -vE "UserWarning|warnings.warn" | tail -9

        "$PY" - "$SET" "$CFG" "$DEC/fidelity.csv" "$CSV" <<'PY'
import csv, sys
setname, cfg, fid, out = sys.argv[1:5]
rows = list(csv.DictReader(open(fid)))
n = len(rows)
deliv = [r for r in rows if r["status"] != "LOST"]
exact = [r for r in rows if r["status"] == "EXACT"]
f1d = [float(r["edge_f1"]) for r in deliv]
f1a = [float(r["edge_f1"]) for r in rows]
ps  = [float(r["psnr_db"]) for r in deliv if r["psnr_db"]]
det = sum(int(r["n_detections"]) for r in deliv)
with open(out, "a", newline="") as f:
    csv.writer(f).writerow([
        setname, cfg, n, len(deliv),
        round(len(deliv)/n, 4) if n else "",
        len(exact),
        round(sum(f1d)/len(f1d), 4) if f1d else 0.0,
        round(sum(f1a)/len(f1a), 4) if f1a else 0.0,
        det,
        round(sum(ps)/len(ps), 2) if ps else "",
    ])
PY
    done
done

echo ""
echo "Results: $CSV"
column -s, -t "$CSV" 2>/dev/null || cat "$CSV"
