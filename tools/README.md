# GBSED over Veins

The GBSED semantic chain with the Sionna MIMO-OFDM channel replaced by a real
VANET: SUMO mobility, IEEE 802.11p, OMNeT++/Veins.

> **Setting this up for the first time, or on another machine?** Read
> `RUNNING.md` in the `gbsed_veins` repository. It covers the OMNeT++/SUMO
> side, the WSL specifics, and the environment variables used below. This file
> documents the two tools themselves.

```
images/                                       gbsed_veins
   │                                              │
   ▼  gbsed_encode.py                             │
scene_data/frame_%04d.bin      ──── staged ──►  GBSEDApp/scene_data/
           frame_%04d.meta.json                    │
           manifest.json                           │  node[0] --802.11p--> node[1]
   │                                               ▼
   │                                        GBSEDApp/received/received_frame_%04d.bin
   ▼  gbsed_decode.py  ◄─────────────────────────  ┘
decoded/fidelity.csv
        png/frame_%04d_reconstructed.png
```

The Python half never runs inside the simulation. The encoder is a
pre-process, the decoder a post-process, and Veins moves opaque bytes in
between — the same shape as `pipeline.py`, where `MIMOE2EModel` sat in the
middle instead.

## Requirements

```bash
pip install numpy torch torchvision opencv-python networkx pandas matplotlib pyyaml pydot
```

Not the full `requirements.txt` — that pulls in TensorFlow, Sionna and a
CUDA 12.6 pin, none of which this path uses.

`roadscene2vec` must sit next to this repo at `../roadscene2vec` (it is not
pip-installed); set `ROADSCENE2VEC_HOME` if yours lives elsewhere. detectron2
is not needed — it is stubbed, and object detection uses torchvision's
Faster R-CNN on CPU, MPS or CUDA.

Two environment variables make the commands below shorter:

```bash
export GBSED_VEINS_APP=/path/to/gbsed_veins/src/veins/modules/application/gbsed/GBSEDApp
export ROADSCENE2VEC_HOME=/path/to/roadscene2vec     # only if not a sibling
```

## 1. Encode

```bash
python tools/gbsed_encode.py \
    --images /path/to/your/pictures \
    --out scene_data \
    --stage "$GBSED_VEINS_APP" \
    --visualize
```

`--stage` defaults to `$GBSED_VEINS_APP`, so it can be omitted once that is
exported.

`--images` takes folders, globs, or individual files. Each frame yields:

| file | transmitted? | contents |
|---|---|---|
| `frame_%04d.bin` | **yes** | the payload — identical to what `GBSED._format_storage_` would have handed the channel model |
| `frame_%04d.meta.json` | no | ground truth: node list, edge list, detections, SHA-256 |
| `manifest.json` | no | frame index, and the `;`-joined `filePath` string for `omnetpp.ini` |

Frames whose graph is only the road/ego/lane skeleton (no actors detected) are
skipped; `--keep-empty` overrides. Other flags: `--device mps`,
`--score-thresh`, `--limit N`, `--config`.

The encoder prints the `omnetpp.ini` line to paste, and warns if the frame
count exceeds the scenario's transmission window (see Limits).

## 2. Transmit

Paste the printed line into `$GBSED_VEINS_APP/omnetpp.ini`:

```ini
*.node[0].appl.filePath = "scene_data/frame_0000.bin;scene_data/frame_0001.bin;..."
```

With 20 frames that line is ~800 characters, so prefer the scripted rewrite in
`RUNNING.md` §5.2 over pasting it by hand. Then:

```bash
cd /path/to/gbsed_veins
./run_gbsed.sh                # Cmdenv
./run_gbsed.sh -u Qtenv       # GUI
```

Received files land in `GBSEDApp/received/received_frame_%04d.bin`.

## 3. Decode and score

```bash
python tools/gbsed_decode.py --meta scene_data --out decoded --visualize
```

`--received` defaults to `$GBSED_VEINS_APP/received`.

Per frame, one of four outcomes:

| status | meaning |
|---|---|
| `EXACT` | reconstructed graph is node- and edge-identical to the original |
| `DEGRADED` | decoded, but the graph differs — scored by edge precision/recall/F1 |
| `CORRUPT` | payload arrived but `format_loading` could not parse it |
| `LOST` | the receiver never completed this file |

`decoded/fidelity.csv` holds one row per frame. `--verbose` prints the full
node/edge diff; a diff is always printed for non-`EXACT` frames.

## Two scenarios

The route file decides whether you are verifying the chain or measuring it.

**Range sweep (current default).** `gbsed.rou.xml` sends `veh0` north and
`veh1` east from the same junction at 8 m/s, so their separation grows as
`sqrt(2)*8*t` — about 100 m at the first chunk and 710 m at the last. The
802.11p range cliff sits near 480 m, so the sequence crosses it mid-run and
you get a fidelity-versus-distance curve. At 8 m/s both vehicles outlive the
whole send queue, so distance is the only variable.

**In-range baseline.** `.orig/gbsed.rou.xml` puts both vehicles on the same
edge at 15 m/s. They stay close, nothing is lost, and every frame comes back
`EXACT`. Use it to prove the chain is lossless before attributing anything to
the channel:

```bash
cd /path/to/gbsed_veins/src/veins/modules/application/gbsed
cp .orig/gbsed.rou.xml GBSEDApp/gbsed.rou.xml     # baseline
git checkout GBSEDApp/gbsed.rou.xml               # back to the sweep
```

## Distance logs

With `writeCsvLog = true` the app appends two files to `outputDir`:

| file | written by | rows |
|---|---|---|
| `tx_log.csv` | sender | one per chunk transmitted |
| `rx_log.csv` | receiver | one per chunk actually heard |

Columns: `fileName, chunkIndex, totalChunks, simTime, txX, txY, rxX, rxY,
distance`. The sender stamps its own position into every `GBSEDMessage`, so
the receiver can compute the separation at reception; `tx_log.csv` leaves the
rx and distance cells empty because the sender does not know where the
receiver is.

`gbsed_decode.py` joins both logs into `fidelity.csv`, adding `tx_time`,
`rx_time`, `distance_m`, `chunks_sent` and `chunks_heard`. That is what lets a
`LOST` frame still report the distance at which its chunks were dropped.

## Limits

**Timing is now configurable.** `startTime` (default 10 s) and `sendInterval`
(default 2 s) are NED parameters, set in `omnetpp.ini`. The number of chunks
you can transmit is `(vehicle_lifetime - startTime) / sendInterval`. The
encoder prints the chunk count and the time of the last send, and warns if it
overruns.

**Shared codebook.** `ACTOR_NAMES` and `RELATION_NAMES` in
`Config/pipeline_extraction.yaml` are index spaces — they are never
transmitted, and node names are re-derived at the receiver from label indices
plus a positional heuristic. Encoder and decoder must load the same yaml. Both
tools hash the two lists into a `codebook` fingerprint and the decoder flags a
mismatch, but pass `--config` consistently.

**No integrity check on the wire.** `GBSEDMessage` carries no checksum, and
`format_loading` ignores the declared length of its last section
(`comp_T = to_read[cur_idx:]`), so a corrupted payload can reshape into a
plausible-but-wrong graph rather than raising. The decoder compares the
received SHA-256 against `meta.json` and reports `payload differs from sent
bytes`, but that is a post-hoc check, not protection.

**All-or-nothing frames — solved by `--format v2`.** With the default v1
layout a single lost chunk loses the frame: the payload is one flat vector and
`format_loading` cannot parse a truncated one.

`--format v2 --chunk-size N` instead packs whole relation slices into blocks
of exactly N bytes, each self-describing, safety relations first. A lost block
costs the relation types it carried. `N` must match `appl.chunkSize` in
`omnetpp.ini`, and must be at least `gbsed_semantic.min_chunk_size()` for the
densest frame (434 B for seq1; slices grow as 2·N²).

Measured on seq1 at 500 B chunks, same delivery, same channel:

| config | safety recall, v1 | safety recall, v2 |
|---|---|---|
| Noise_Low | 0.80 | **1.00** |
| Noise_Medium | 0.50 | **1.00** |

Cost: zero extra chunks on the wire at 1000 B, one extra at 500 B. At 1000 B
the formats are indistinguishable on this sequence, because only 2 of 20
frames span more than one chunk.

Requires `writePartialFiles` (default true) so the receiver hands over
incomplete files. Note that this also makes a truncated **v1** payload parse
without complaint into a silently wrong graph — prefer v2 when partial writing
is on.

## Module layout

`gbsed_semantic.py` (repo root) holds the shared chain — detectron2 stub,
`LiteExtractor`, config loading, the four serialization functions, the
detector, and the graph comparison helpers. It is the single copy of code that
was previously duplicated between `load_model.ipynb` cells 10–14 and
`pipeline/pipeline.py:97-227`. Both CLIs import it; the notebook can too.
