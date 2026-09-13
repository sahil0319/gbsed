# Project overview — everything, start to finish

What this project is, what we built, every experiment we ran, what we found,
and where all of it lives. Plain language; no prior context assumed.

> Twelve markdown files exist across the two repositories. **This is the one to
> read first.** A map of the rest is in §10.

---

## 1. The project in one page

**The problem.** Self-driving cars would be safer if they told each other what
they see. But a single camera frame is about 1.2 MB, and the radio link between
two passing vehicles carries roughly 6 Mbit/s *shared between everyone in
range*, for only a few tens of seconds. Sending pictures does not work.

**The idea.** Send the *meaning* of the scene instead of the pixels. Convert the
image into a **scene graph** — a small network of nodes (the ego car, the lanes,
the road, each detected vehicle) joined by labelled edges (`isIn`,
`toLeftOf`, `near_coll`) — compress it, transmit that, and rebuild it at the
other end. This is called *semantic communication*.

**What already existed.** A published framework, **GBSED** (Ribouh & Di Ngoma,
2026), which does exactly this. Its authors report 99.9% compression against
raw images and good fidelity at moderate signal strength.

**What was missing, and what we did.** GBSED was tested against an *abstract
radio channel*: a signal-to-noise sweep with no moving vehicles, no packet loss,
no network. And its headline compression figure was measured against *raw
pixels*, which nobody actually transmits.

We did two things:

1. **Put GBSED on a real vehicular network** — OMNeT++, Veins and SUMO over
   IEEE 802.11p, with vehicles that move, separate, and eventually lose contact.
2. **Built a fair comparison.** We re-encoded each image down to *exactly the
   same number of bytes* as its scene graph, so identical packets cross an
   identical radio and the only thing that differs is what those bytes mean.

**The headline answer.** At the same byte budget, the scene graph recovers
**44 of 44** safety-critical relations; the image recovers **0 of 44**. Images
need roughly **96× more bandwidth** to catch up.

---

## 2. How the system works

```
  ROAD IMAGE (1280×720)
        │
        │  object detector finds cars, people, signs
        ▼
  SCENE GRAPH          nodes = road, ego car, 3 lanes, each detected actor
        │              edges = relations (isIn, near_coll, toLeftOf, …)
        │
        │  encode():  adjacency tensor T [12 relations × N × N]  +  features F
        ▼
  SEMANTIC COMPRESSION    throw away relation types absent from this scene
        │                 (typically drops 4–8 of the 12)
        │
        │  serialise to float16 bytes  →  ~800 bytes per frame
        ▼
  ┌─────────────────────────────────────────────┐
  │  THE NETWORK  (what this project added)     │
  │  chunk → Base64 → IEEE 802.11p broadcast    │
  │  two SUMO vehicles, moving apart            │
  └─────────────────────────────────────────────┘
        │
        ▼
  DESERIALISE → SEMANTIC DECOMPRESSION → SCENE GRAPH′ → risk classifier
```

**One thing that surprises people:** node names are *never transmitted*. The
receiver re-derives "car_0", "Left Lane" and so on from numeric label indices
plus a positional rule. Both ends must agree on the same list of actor and
relation names — a "shared codebook". This saves bytes but means reordering
either list silently corrupts every previously encoded payload.

---

## 3. What we built, phase by phase

### Phase 1 — Get the semantic layer running standalone

The published code imports TensorFlow, Sionna and detectron2 — a heavy,
GPU-specific stack, most of which is only needed for the radio simulation we
were replacing anyway.

- Consolidated the semantic chain into one importable module,
  `gbsed_semantic.py`. The serialisation functions had been **duplicated
  verbatim** between the pipeline script and an exploratory notebook — an
  arrangement guaranteed to drift apart.
- Stubbed out detectron2 (which requires CUDA) and substituted torchvision's
  Faster R-CNN, which produces the same output and runs on CPU.
- Result: the encode → decode round trip works with numpy and torch alone.

### Phase 2 — Bridge Python to the network simulator

- `tools/gbsed_encode.py`: a folder of images → one `.bin` payload per frame,
  plus a `.meta.json` recording the true graph (ground truth for scoring) and a
  manifest.
- `tools/gbsed_decode.py`: received bytes → rebuilt scene graph → scored
  against that ground truth.
- The Veins application already moved files byte-exactly, so no C++ changes
  were needed yet.

**First end-to-end run:** 20 frames, all delivered, all reconstructed perfectly.

### Phase 3 — Make the simulation actually test something

The first result was 100% delivery — because both vehicles drove the *same
road at the same speed* and never moved apart. The simulation was measuring
nothing.

Changes to the Veins application (C++):

| change | why |
|---|---|
| `startTime` / `sendInterval` became parameters | they were hard-coded, so the transmittable volume was a property of the source code rather than the experiment |
| sender stamps its position into every message | without it, a lost packet has no distance attached and delivery cannot be analysed against range |
| per-chunk `tx_log.csv` / `rx_log.csv` | this is what makes the range analysis possible at all |
| vehicles now diverge at right angles at 8 m/s | so separation grows continuously and the sequence crosses the radio range limit mid-run |

**Why 8 m/s specifically:** at the original 15 m/s the vehicles are removed from
the simulation at t≈67 s, *before* the send queue finishes — which would confuse
"lost to distance" with "the sender vanished". At 8 m/s they outlive the queue,
so distance is the only variable.

**Result:** 14 of 20 frames delivered. Measured range cliff at **488 m**.

### Phase 4 — The channel sweep

Five radio conditions (noise floor from −98 to −80 dBm) × two driving
sequences, each decoded and fed to the downstream risk classifier.

### Phase 5 — The pixel baseline (the core contribution)

Three ways to send images instead of graphs:

- **Full image** — the original JPEG as-is. Answers a feasibility question.
- **Matched budget** — each image squeezed to *that frame's exact* GBSED byte
  count, by searching over resolution and quality. Identical bytes ⇒ identical
  packets ⇒ identical delivery. **This is the controlled experiment.**
- **Budget sweep** — the budget multiplied by 1, 2, 4 … 128 with the channel
  removed, producing a rate-versus-meaning curve.

### Phase 6 — Graceful degradation (slice-aligned packing)

We noticed the decoder *already* tolerated a partial relation set — semantic
decompression writes each delivered slice into its slot and leaves the rest
zero. Only the byte layout threw that away: one flat vector, so losing any chunk
destroyed the frame.

The **slice-aligned layout** packs whole relation slices into self-describing
blocks the same size as a network chunk, safety relations first. A lost block
now costs a relation *type*, not the frame. The receiver also had to be changed
to hand over incomplete files, which it previously discarded. (In the encoder
CLI this layout is selected with `--format v2`; the original flat layout is
`--format v1`.)

---

## 4. Results

### 4.1 The semantic chain survives a real network

Every delivered frame reconstructed **exactly** — same nodes, same edges — in
every configuration, across both sequences.

This is a stronger statement than the original SNR study can make. 802.11p
either delivers a frame with a valid checksum or drops it, so there is no
half-corrupted payload. **In a VANET, semantic degradation is about which
frames arrive, not how damaged they are.**

### 4.2 Fixed budget: the central result

Same bytes, same channel, same frames — only the representation differs.

| arm | delivered | graph recovered exactly | safety relations kept |
|---|---|---|---|
| **GBSED (scene graph)** | 70% | **70%** | **32/32** |
| WebP at same bytes | 70% | **0%** | **0/32** |
| JPEG at same bytes | 70% | **0%** | **0/32** |

At that budget an image is **102×58 to 256×144 pixels at quality 1–5**. Across
the 14 delivered frames the detector found 18 objects in the WebP versions and
**zero** in the JPEG versions, against roughly 40 in the originals.

WebP was given every advantage — it beats JPEG at low bitrate, came in *under*
budget, and held 256×144 where JPEG collapsed. It still recovered nothing.

### 4.3 How much more bandwidth would images need?

| budget | bytes/frame | WebP safety kept | JPEG safety kept |
|---|---|---|---|
| ×1 | 800 | 0/44 | 0/44 |
| ×4 | 2,900 | 16/44 | 0/44 |
| ×16 | 12,100 | 34/44 | 2/44 |
| ×64 | 43,500 | 42/44 | 40/44 |
| ×128 | 83,000 | **44/44** | 42/44 |
| **GBSED** | **811** | **44/44** | — |

**WebP first matches GBSED at 77,809 bytes/frame — 96×.** JPEG never does
within the sweep.

Efficiency: GBSED spends **369 bytes per preserved safety relation**; WebP at
the budget where it finally matches spends **35,368**.

**The built-in validity check.** The sweep includes the *original* image at
unbounded budget. Because the ground truth was generated from those images, the
image arm **must** score 1.000 there — and it does. That is what proves the gap
is a bandwidth effect and not a rigged image pipeline.

### 4.4 Sending whole frames is impossible, not just wasteful

| | GBSED | full JPEG |
|---|---|---|
| 20 frames | 16,218 B | 24,839,399 B (**1531×**) |
| time to transmit | 55 s | 62,118 s (**17.3 hours**) |
| vehicles in range for | 125 s | 125 s |
| frames delivered | 14/20 | **0/20** |

Measured, not extrapolated: **1.24% of the first frame arrived.**

### 4.5 Graceful degradation works

At 500-byte chunks, same bytes and same delivery:

| condition | flat layout | **slice-aligned** |
|---|---|---|
| Noise_Low | 0.80 | **1.00** |
| Noise_Medium | 0.50 | **1.00** |

Cost: **zero extra chunks** on the air at 1000 B, one extra at 500 B.

One frame shows the mechanism. It got 1 of its 2 chunks under both formats:

```
flat layout     safety 0/2   "payload differs from sent bytes"
slice-aligned   safety 2/2   "partial: 1 block lost, 3/7 relations recovered"
```

Same chunk lost. The flat layout loses both safety relations; the slice-aligned layout keeps both **and says what
is missing**.

---

## 5. Why it works — three separate mechanisms

Usually collapsed into "compression"; they are not the same thing.

**1. It compresses the right thing.** A JPEG spends bits on texture, lighting,
road surface and sky — none of which the risk classifier reads. The scene graph
spends them on who is where and how they relate, which is all it reads.

**2. It degrades in a different dimension.** Image codecs blur everything
uniformly, so below a threshold the detector fails on *everything at once* —
that is the cliff in the sweep, where JPEG sits at exactly 0.000 through ×8 and
then rises steeply. A scene graph splits into independent relation slices, so
losing bits costs relation categories instead.

**3. Semantic content is discrete.** A relation is a yes/no edge; a label is a
small integer. There is no "slightly wrong `near_coll`". This is why every
delivered frame scores exactly 1.000 and never 0.98.

**The honest framing.** GBSED is *not better compression*. It transmits the
**output** of perception rather than its **input**, and wins because the
receiver's task needs only that output. The cost: the receiver can never do
anything the sender's detector did not already encode. A raw image is
task-agnostic; a scene graph is committed. The comparison is fair only because
the downstream task was fixed in advance.

---

## 6. Problems we found and fixed

All three would have distorted the conclusions.

**Three of four fidelity metrics were the same number.** Delivery rate,
bit-exact rate and mean edge F1 were identical in every row. Two causes:
bit-exact ≡ delivery is *real* (802.11p never delivers a partly corrupt frame),
but mean edge F1 ≡ delivery was a **bug** — the code filtered lost frames with
`if r["edge_f1"]`, and their value is the *string* `"0.0"`, which Python treats
as true. Lost frames entered the average as zeros, collapsing the metric.

**Edge F1 had a free floor of 0.38.** Every graph contains the road, ego and
three lanes joined by `isIn` edges regardless of image content — about 38% of a
typical edge set. A receiver decoding a *blank image* scored 0.386. We added
**actor-edge F1**, which ignores that skeleton and reads 0.000 for the same
input.

**Three of eight channel configs were exact duplicates.** They set a mobility
parameter that TraCI overwrites every update. The proof is bit-identical
classifier output — `0.9753320813179016` for both `Baseline` and `CAV_Good` —
which can only come from identical inputs. Eight configs were really five noise
floors. This also biased the accuracy summary, since duplicated conditions were
counted twice.

We also added **safety-relation recall** as the headline metric, and
`n_frames_classified` so a flipped prediction can be told apart from the
classifier simply seeing fewer frames.

---

## 7. Figures — what each one shows

### Current

| figure | shows | what to look for |
|---|---|---|
| `budget_sweep/01_rate_semantics.png` | **the headline.** Bytes/frame (log) vs meaning recovered | GBSED is the green star at 811 B; the image curves need ~10⁵ bytes to reach it. The dotted line is the validity control. |
| `comparison/01_gbsed_vs_image.png` | fixed budget over the real channel | the three *overlapping* lines in the left panel **are the point** — the control held. The other two panels diverge completely. |
| `comparison/02_payload_size.png` | payload sizes, log scale | the 1531× gap |
| `experiment_results_format/01_format_sweep.png` | flat vs slice-aligned packing under partial loss | the 1000 B pair *coinciding* is expected, not a null result — at that size only 2 of 20 frames span multiple chunks |
| `report/figures/architecture.png` | the full pipeline | green = transmitter, blue = our network work, red = receiver |
| `report/figures/scenario.png` | SUMO geometry + measured delivery vs distance | separation grows at 10.35 m/s; last chunk heard at 488 m |
| `report/slides_figures/loss_resilience.png` | one frame, one lost chunk | the collision-risk relation survives in the reconstructed graph; a flat layout would have lost it at the same edge count |
| `scene_data_seq1/png/` vs `decoded_seq1/png/` | per-frame graphs, sent vs received | identical for every delivered frame |

### Superseded — regenerate before using

All seven `result_graphs/*.png` predate the fixes and plot the eight-config set,
so all carry the duplicate artefact.

| figure | problem |
|---|---|
| `01_delivery_rate_by_config.png` | sawtooth: `CAV_Good` appears to recover, but it is `Baseline` replotted |
| `02_bit_exact_rate_by_config.png` | redundant — identical to 01 |
| `03_fidelity_metrics_overview.png` | **three panels, one curve.** Kept in the report as a "before" exhibit |
| `04_prediction_correctness_heatmap.png` | duplicate columns; `CAV_Extreme` missing (the sweep aborted on it) |
| `05_true_class_confidence_trend.png` | duplicate columns; the non-monotonic tail is real but confounded by frame count |
| `06_accuracy_summary.png` | **biased** — duplicated conditions counted twice (seq1 reads 3/7 = 0.43 where the true value is 2/4 = 0.50) |
| `07_fidelity_vs_confidence_scatter.png` | duplicate points overlap, overstating the number of observations |

---

## 8. Data files

| file | contents |
|---|---|
| `budget_sweep/budget_sweep.csv` | the rate–meaning curve |
| `comparison/comparison.csv` | matched budget over the channel |
| `experiment_results_format/format_results.csv` | flat vs slice-aligned |
| `experiment_results/results_matrix.csv` | the GBSED channel sweep |
| `experiment_results_image/image_results_matrix.csv` | the pixel arms |
| `*/decoded_*/fidelity.csv` | per-frame detail for every run |
| `*/received_*/{tx,rx}_log.csv` | per-chunk logs with distance |

Useful per-frame columns: `status` (EXACT / DEGRADED / CORRUPT / LOST),
`actor_edge_f1`, `risky_preserved`/`risky_orig`, `distance_m`, and for the slice-aligned layout
`blocks_missing` / `relations_recovered`.

---

## 9. Running everything

Set once:

```bash
export GBSED_VEINS_APP=~/Documents/gbsed_veins/src/veins/modules/application/gbsed/GBSEDApp
```

**Normal run** (images → transmit → score):

```bash
# 1. clear old artefacts (frames are numbered and never deleted)
rm -rf scene_data decoded "$GBSED_VEINS_APP/scene_data" "$GBSED_VEINS_APP/received"

# 2. encode
python tools/gbsed_encode.py --images images_seq1 --out scene_data --visualize

# 3. point the simulation at them (writes the filePath line from the manifest)
python - <<'PY'
import json, os, re, pathlib
p = pathlib.Path(os.environ["GBSED_VEINS_APP"]) / "omnetpp.ini"
fp = json.load(open("scene_data/manifest.json"))["omnetpp_filePath"]
p.write_text(re.sub(r'^\*\.node\[0\]\.appl\.filePath = .*$',
                    '*.node[0].appl.filePath = "%s"' % fp, p.read_text(), flags=re.M))
PY

# 4. simulate   (add -u Qtenv for the GUI)
cd ~/Documents/gbsed_veins && ./run_gbsed.sh

# 5. decode and score
cd - && python tools/gbsed_decode.py --meta scene_data --out decoded --visualize
```

**Reproduce the experiments:**

```bash
python tools/budget_sweep.py --meta scene_data_seq1 --images images_seq1 --out budget_sweep
python tools/image_encode.py --meta scene_data_seq1 --images images_seq1 --mode matched --codec webp --out image_data_seq1_webp
python tools/image_encode.py --meta scene_data_seq1 --images images_seq1 --mode matched --codec jpeg --out image_data_seq1_jpeg
./tools/run_image_baseline.sh image_data_seq1_webp image_data_seq1_jpeg
python tools/compare_gbsed_vs_image.py --out comparison

python tools/gbsed_encode.py --images images_seq1 --out scene_data_seq1_v2_cs500 --format v2 --chunk-size 500
./tools/run_format_sweep.sh && python tools/plot_format_sweep.py
```

**Build the report:**

```bash
cd report && latexmk -pdf main.tex
```

---

## 10. Map of the documentation

Paths below are relative to this repository, except those beginning
`gbsed_veins/`, which live in the **Veins repository**. The documented layout
puts the two side by side; on the development machine they are at
`~/Desktop/PythonEnvs/gbsed` and `~/Documents/gbsed_veins`.

| file | for | read when |
|---|---|---|
| **`overview.md`** (this file) | everyone | **first** |
| `comparison/ANALYSIS.md` | the full method and findings | you need the detail behind §4–6 |
| `report/main.pdf` | the graded deliverable | submission |
| `report/PRESENTATION.md` | slide content, timing, Q&A prep | before the defence |
| `report/README.md` | how to build the report, what to fill in | before submission |
| `tools/README.md` | the encoder and decoder CLIs | using the tools |
| `gbsed_veins/RUNNING.md` | setup on a new machine, WSL notes | onboarding a teammate |
| `gbsed_veins/overview.md` | codebase orientation, invariants | an AI assistant, or deep code work |
| `gbsed_veins/GBSED.md` | macOS/opp_env build history | toolchain trouble |
| `CLAUDE.md` (both repos) | auto-loaded pointers | — |
| `READING_GUIDE.md`, `README.md` | inherited from upstream | background on the original code |

---

## 11. Still open

1. **No negative class.** Both sequences are labelled risky, so accuracy cannot
   distinguish a working classifier from one that always says "risky". This is
   the single biggest limitation of the task-level results.
2. **`result_graphs/` needs regenerating** against the deduplicated configs and
   corrected metrics.
3. **The slice-aligned layout is not the default** — the flat serialisation
   still is, to stay byte-identical to the published implementation. With
   partial-file writing enabled a truncated flat payload parses into a silently
   wrong graph, so the slice-aligned layout should be preferred there.
   (CLI: `--format v1` flat, `--format v2` slice-aligned.)
4. **The slice-aligned benefit needs small chunks or denser scenes** to show at 1000 B.
5. **Payloads reproduce as graphs, not bytes.** Re-encoding on another platform
   gave 17/20 byte-identical payloads, the rest differing by one unit in the
   last place of a single float16 feature. All 20 graphs were identical.
6. **Two vehicles only** — no medium-access contention, which is where 802.11p
   gets genuinely hard.

---

## 12. Repositories

- `https://github.com/sahil0319/gbsed` — semantic layer, tooling, experiments,
  report
- `https://github.com/Loona6/gbsed_veins` — Veins application and simulation
  scenario
