#!/usr/bin/env bash
#
# run_format_sweep.sh -- does slice-aligned packing (v2) actually buy graceful
# degradation over the flat layout (v1)?
#
# Sweeps (format x chunkSize x channel config). chunkSize matters: at 1000 B
# nearly every frame in this sequence is a single chunk, and a single-chunk
# frame has nothing to reorder -- it arrives whole or not at all. The effect
# only has room to appear once frames span several chunks.
#
set -uo pipefail

GBSED_REPO="${GBSED_REPO:-$HOME/Desktop/PythonEnvs/gbsed}"
VEINS_REPO="${VEINS_REPO:-$HOME/Documents/gbsed_veins}"
APP="$VEINS_REPO/src/veins/modules/application/gbsed/GBSEDApp"
PY="${PY:-$HOME/miniconda3/envs/av/bin/python}"
RESULTS="$GBSED_REPO/experiment_results_format"
CONFIGS=(Baseline Noise_Low Noise_Medium Noise_High CAV_Extreme)

# "label:scene_data_dir:chunkSize"
ARMS=(
  "v1_cs1000:scene_data_seq1:1000"
  "v1_cs500:scene_data_seq1:500"
  "v2_cs1000:scene_data_seq1_v2_cs1000:1000"
  "v2_cs500:scene_data_seq1_v2_cs500:500"
)

mkdir -p "$RESULTS"
CSV="$RESULTS/format_results.csv"
echo "arm,format,chunk_size,config,frames,delivered,partial,graph_exact,mean_edge_f1_delivered,mean_actor_f1_delivered,risky_total,risky_preserved,risky_recall,chunks_sent" > "$CSV"

for ARM in "${ARMS[@]}"; do
    IFS=: read -r LABEL SETDIR CS <<< "$ARM"
    FMT="${LABEL%%_*}"
    echo ""
    echo ">> staging $LABEL  ($SETDIR, chunkSize=$CS)"
    rm -rf "$APP/scene_data"; mkdir -p "$APP/scene_data"
    cp "$GBSED_REPO/$SETDIR"/*.bin "$APP/scene_data/"
    "$PY" - "$GBSED_REPO/$SETDIR" "$APP/omnetpp.ini" <<'PY'
import json, re, sys, pathlib
setdir, ini = sys.argv[1], pathlib.Path(sys.argv[2])
fp = json.load(open(f"{setdir}/manifest.json"))["omnetpp_filePath"]
ini.write_text(re.sub(r'^\*\.node\[0\]\.appl\.filePath = .*$',
                      '*.node[0].appl.filePath = "%s"' % fp,
                      ini.read_text(), flags=re.M))
PY

    for CFG in "${CONFIGS[@]}"; do
        printf "   %-12s %-13s " "$LABEL" "$CFG"
        rm -rf "$APP/received"; mkdir -p "$APP/received"
        ( cd "$VEINS_REPO" && ./run_gbsed.sh -c "$CFG" -u Cmdenv \
            "--*.node[*].appl.chunkSize=$CS" ) >/dev/null 2>&1

        ARCH="$RESULTS/received_${LABEL}_${CFG}"
        rm -rf "$ARCH"; cp -r "$APP/received" "$ARCH"
        DEC="$RESULTS/decoded_${LABEL}_${CFG}"

        "$PY" "$GBSED_REPO/tools/gbsed_decode.py" --received "$ARCH" \
            --meta "$GBSED_REPO/$SETDIR" --out "$DEC" >/dev/null 2>&1

        "$PY" - "$LABEL" "$FMT" "$CS" "$CFG" "$DEC/fidelity.csv" "$ARCH" "$CSV" <<'PY'
import csv, os, sys
label, fmt, cs, cfg, fid, arch, out = sys.argv[1:8]
rows = list(csv.DictReader(open(fid)))
def col(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else 0.0
n = len(rows)
deliv = [r for r in rows if r["status"] != "LOST"]
exact = [r for r in rows if r["status"] == "EXACT"]
partial = [r for r in deliv if int(col(r, "blocks_missing")) > 0]
f1 = [col(r, "edge_f1") for r in deliv]
af1 = [col(r, "actor_edge_f1") for r in deliv]
rt = sum(int(col(r, "risky_orig")) for r in deliv)
rk = sum(int(col(r, "risky_preserved")) for r in deliv)
tx = os.path.join(arch, "tx_log.csv")
sent = sum(1 for _ in open(tx)) - 1 if os.path.isfile(tx) else ""
with open(out, "a", newline="") as f:
    csv.writer(f).writerow([
        label, fmt, cs, cfg, n, len(deliv), len(partial), len(exact),
        round(sum(f1)/len(f1), 4) if f1 else 0.0,
        round(sum(af1)/len(af1), 4) if af1 else 0.0,
        rt, rk, round(rk/rt, 4) if rt else "", sent])
print("deliv=%2d (partial %2d) exact=%2d actorF1=%.3f safety=%2d/%-2d"
      % (len(deliv), len(partial), len(exact),
         sum(af1)/len(af1) if af1 else 0.0, rk, rt))
PY
    done
done
echo ""
column -s, -t "$CSV" 2>/dev/null || cat "$CSV"
