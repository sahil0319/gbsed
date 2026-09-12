# GBSED vs. transmitting the image, at a fixed bit budget

**Sequence:** seq1, 20 frames. **Channel:** the five distinct conditions in
`omnetpp.ini`. **Reproducibility:** the GBSED Baseline run was repeated on a
second machine (macOS/opp_env) before any comparison and returned 14/20
delivered, identical to the Linux result.

---

## 1. The headline result

**At the same byte budget, pixels recover nothing. To match GBSED, WebP needs
96× more bandwidth and JPEG more than 109×.**

| | bytes/frame | graph exact | safety relations | actor-edge F1 |
|---|---|---|---|---|
| **GBSED** | **811** | **20/20** | **44/44** | **1.000** |
| WebP @ same budget | 730 | 0/20 | 0/44 | 0.000 |
| JPEG @ same budget | 812 | 0/20 | 0/44 | 0.000 |
| WebP, 96× budget | 77,809 | 14/20 | 44/44 | 0.974 |
| Original frame (unbounded) | 1,241,970 | 20/20 | 44/44 | 1.000 |

The last row is the experiment's own sanity check. The ground truth was
produced from the original frames, so an unbounded budget *must* score 1.000.
It does — which means the gap above is a bitrate effect, not a broken image
pipeline.

---

## 2. Two experiments

The obvious comparison — "send the JPEG instead" — answers a feasibility
question. It is reported in §4, but it is not the argument.

### 2.1 Matched budget over the real channel (§3)

Each image is re-encoded to *that frame's exact GBSED byte budget*. Same
bytes → same chunk count → the same packets over the same radio. Measured
delivery is identical to GBSED at every condition, which confirms the control
held. Everything that differs afterwards is the representation alone.

### 2.2 Budget sweep, channel-free (§5)

The budget is then swept upward to find where pixels catch up. Deliberately
channel-free, so the curve measures representation efficiency alone. Putting
the channel back only hurts the image arm further, since larger payloads span
more chunks and one lost chunk loses the frame.

Both arms run the **same** detector, the **same** roadscene2vec extraction,
and are scored against the **same** ground-truth graphs by the **same** code
(`gbsed_semantic.edge_metrics`). The received image is upscaled to 1280×720
before detection — not a handicap but a requirement, since the BEV homography
and the extractor's distance thresholds are calibrated for that size.

---

## 3. Matched budget, over the channel

| arm | config | delivered | graph exact | safety kept | edge F1 | **actor F1** |
|---|---|---|---|---|---|---|
| GBSED | Baseline | 70% | **70%** | **32/32** | 1.000 | **1.000** |
| GBSED | Noise_Low | 55% | 55% | 24/24 | 1.000 | 1.000 |
| GBSED | Noise_Medium | 25% | 25% | 8/8 | 1.000 | 1.000 |
| WebP @ budget | Baseline | 70% | **0%** | **0/32** | 0.378 | **0.000** |
| WebP @ budget | Noise_Low | 55% | 0% | 0/24 | 0.381 | 0.000 |
| JPEG @ budget | Baseline | 70% | **0%** | **0/32** | 0.386 | **0.000** |

At this budget an image is 102×58 to 256×144 pixels at quality 1–5. Across the
14 frames delivered on the best channel the detector found 18 objects in the
WebP frames and **zero** in the JPEG frames, against ~40 in the originals.

WebP was given every advantage — better at low bitrate, came in *under* budget
(14,604 B vs GBSED's 16,218 B), held 256×144 where JPEG collapsed to 102×58.
It still recovered nothing.

Note the two F1 columns. Plain edge F1 says 0.378; actor-edge F1 says 0.000.
The difference is the skeleton — see §6.2.

---

## 4. Sending the whole frame

| | GBSED | full JPEG |
|---|---|---|
| total bytes, 20 frames | 16,218 | 24,839,399 (**1531×**) |
| chunks @1000 B | 22 | 24,847 |
| simulated time to send | 55 s | 62,118 s (**17.3 hours**) |
| sender vehicle lifetime | 125 s | 125 s |
| frames delivered (Baseline) | 14/20 | **0/20** |

Measured, not extrapolated: the receiver heard 16 chunks, all belonging to
frame 0, which needs 1,295. **1.24% of one frame.**

---

## 5. The rate–semantics curve

`budget_sweep/01_rate_semantics.png`. Channel-free; x-axis is bytes per frame.

| budget | bytes/frame | WebP safety | WebP actor F1 | JPEG safety | JPEG actor F1 |
|---|---|---|---|---|---|
| ×1 | ~800 | 0/44 | 0.000 | 0/44 | 0.000 |
| ×2 | ~1,500 | 6/44 | 0.084 | 0/44 | 0.000 |
| ×4 | ~2,900 | 16/44 | 0.327 | 0/44 | 0.000 |
| ×8 | ~6,100 | 24/44 | 0.511 | 0/44 | 0.000 |
| ×16 | ~12,100 | 34/44 | 0.774 | 2/44 | 0.025 |
| ×32 | ~23,700 | 42/44 | 0.885 | 14/44 | 0.389 |
| ×64 | ~43,500 | 42/44 | 0.916 | 40/44 | 0.860 |
| ×128 | ~83,000 | **44/44** | 0.974 | 42/44 | 0.901 |
| **GBSED** | **811** | **44/44** | **1.000** | — | — |

**WebP first matches GBSED's safety recall at 77,809 B/frame — 96×.** JPEG
never does within the sweep: 42/44 at 88,383 B/frame, 109×.

Efficiency, stated as bytes per preserved safety relation:

| | total bytes | safety kept | **bytes per relation** |
|---|---|---|---|
| GBSED | 16,218 | 44 | **369** |
| WebP ×128 | 1,556,180 | 44 | 35,368 (**96×**) |
| WebP ×1 | 14,604 | 0 | undefined |

---

## 6. Why GBSED works

Three distinct mechanisms, worth separating because they are usually collapsed
into "compression".

**1. It compresses the right thing.** A JPEG spends bits on texture, lighting,
road surface, sky — none of which the downstream task reads. The scene graph
spends them on actor identity, position and pairwise relations, which is
exactly what the classifier consumes. The 1531× reduction is not a better
codec; it is a change in what counts as signal.

**2. Degradation is graceful in the right dimension.** Image codecs degrade
*spatially and uniformly*: every object blurs together, and below a threshold
the detector fails on all of them at once. That threshold is visible in §5 —
JPEG sits at exactly 0.000 through ×8 and then rises steeply. The scene graph
degrades *by relation type* instead, because `sem_compression` already
decomposes it into independent slices. (Currently unexploited — §7.5.)

**3. Semantic content is discrete, so it survives quantisation.** A relation is
a binary edge; node labels are small integers. These are exactly representable
in float16 and are either right or wrong — there is no "slightly wrong
`near_coll`". Pixels are continuous and every bit removed degrades them. That
is why every delivered GBSED frame scores exactly 1.000 and never 0.98.

**The honest framing.** GBSED is not better compression. It transmits the
*output* of perception rather than its *input*, and wins because the receiver's
task needs only that output. The corollary is the limitation: the receiver can
never do anything the sender's extractor did not already encode. A raw image is
task-agnostic; a scene graph is committed. The comparison is fair only because
the downstream task was fixed in advance. That caveat belongs in the paper.

---

## 7. Issues found and fixed

### 7.1 Three of eight configs were exact duplicates — FIXED

`CAV_Good`, `CAV_Moderate` and `CAV_Bad` differed from `Baseline`,
`Noise_Medium` and `Noise_High` only by `*.node[1].veinsmobility.x`. That
parameter has **no effect**: node positions come from SUMO over TraCI, which
overwrites it every update.

The evidence was bit-identical classifier output, not merely similar metrics:

```
Baseline     prob_class1 = 0.9753320813179016
CAV_Good     prob_class1 = 0.9753320813179016   <- same float, last bit
Noise_Medium / CAV_Moderate = 0.056105922907590866
Noise_High   / CAV_Bad      = 0.20187801122665405
```

Removed from `omnetpp.ini`; `CONFIGS` in the matrix script reduced to the five
real noise floors (−98, −95, −90, −85, −80 dBm). `CAV_Extreme` keeps its name
so the archived result directories stay valid, with a comment noting it is the
−80 dBm point and has nothing to do with mobility.

Worth keeping in mind: distance *is* already swept continuously inside every
run, because the two vehicles diverge. A distance config would have to change
the route, not a mobility offset.

### 7.2 CAV_Extreme ran but was never logged — FIXED

`decoded_seq1_CAV_Extreme/` existed with no corresponding CSV row.
`set -euo pipefail` plus a zero-delivery run made the risk-assessment step exit
non-zero and kill the sweep. (It also explains why seq2's rows precede seq1's —
the first pass died and was restarted.) The per-run body no longer uses `set -e`,
and a missing prediction is logged as an empty field rather than aborting.

### 7.3 `mean_edge_f1` was algebraically identical to `delivery_rate` — FIXED

Every row of the old `results_matrix.csv` had
`delivery_rate == bit_exact_rate == mean_edge_f1`. Two causes:

- `bit_exact_rate ≡ delivery_rate` is **real**: 802.11p hands up a frame with a
  valid FCS or drops it, so a delivered frame is always bit-exact. Kept as an
  assertion that should never fire.
- `mean_edge_f1 ≡ delivery_rate` was a **bug**. The filter
  `[float(r["edge_f1"]) for r in rows if r["edge_f1"]]` meant to skip LOST
  frames, but their `edge_f1` is the *string* `"0.0"`, which is truthy. They
  entered as zeros, and since delivered frames always score 1.0 the mean
  collapsed to the delivery rate.

Now reported as two explicitly named columns, `mean_edge_f1_delivered` and
`mean_edge_f1_all`.

### 7.4 Edge F1 had a free floor of ≈0.38 — FIXED

The JPEG arm scored 0.386 mean edge F1 **with zero detections**: every graph
contains `Root Road`, `ego car` and three lanes joined by `isIn` edges
regardless of image content, and that skeleton is ~38% of the edge set. Edge F1
therefore ranged over [0.38, 1], with the bottom meaning "understood nothing".

Added `actor_edge_f1`, scored only over edges with a detected actor at one end.
It reads **0.000** for every image arm at matched budget — the discrimination
the old metric could not make.

### 7.5 Per-relation-slice chunking — IMPLEMENTED (see §10)

Mechanism 2 in §6 was claimed by the architecture but never exercised: one
lost chunk lost the whole frame. Now implemented as payload format **v2**, and
measured. At 500 B chunks, where frames actually span blocks, safety-relation
recall under partial delivery goes from **0.80 / 0.50** (v1) to **1.00 / 1.00**
(v2) at Noise_Low / Noise_Medium, with identical bytes and identical delivery.

### 7.6 Still open: no negative class

Both sequences are labelled risky, so `match` cannot distinguish a working
classifier from one that always predicts 1, and no false-positive rate is
computable. Set aside for now by agreement, but it bounds what the task-level
numbers can claim.

### 7.7 Minor: payloads are not byte-reproducible across platforms

Re-encoding seq1 on macOS/arm64 reproduced 17 of 20 payloads byte-identically
to the Linux-encoded reference. The other three differ by **one ULP in a single
float16 feature value** (e.g. `-0.061370849609375` vs `-0.06134033203125`) —
cross-platform floating point in the detector or the BEV projection.

The extracted **graphs are identical in all 20 frames**, so nothing downstream
is affected. But it means the encoder is reproducible at the level of scene
graphs, not of bytes. Anything that compares payload hashes across machines
(the decoder's `bit_exact` column does, against `meta.json`) must re-encode and
re-hash on the same machine, or compare graphs instead.

### 7.8 Minor: duplicate frames in seq1

`images_seq1/` contains both `.jpg` and `.png` of `00097100` and `00097114`, so
frames 3/4 and 18/19 are the same scene. Harmless, but it inflates the frame
count by two.

---

## 10. Slice-aligned packing (format v2)

### 10.1 What was wrong

`format_storage()` concatenates everything into one flat float16 vector, so
the transport's fixed-size byte chunking cuts through the middle of relation
slices. Lose a chunk and `format_loading()` runs off the end of the buffer:
the frame is discarded.

But `sem_decompression()` **already tolerates a partial relation set** — it
writes each delivered slice to its index in `T` and leaves the rest zero, and
cannot distinguish "this relation was absent from the scene" from "this
relation's slice did not arrive". The capability was there; only the packing
threw it away.

### 10.2 What v2 does

The payload is built as a whole number of blocks of exactly `chunkSize` bytes,
each self-describing (magic `GBS2`, node count, feature width, slice count)
and holding only **complete** relation slices. The transport still chunks
naively at `chunkSize`, so every network chunk is one self-contained block.
Block 0 also carries the node block; slices are ordered safety-first.

A missing block is all-zero — the receiver zero-fills its buffer — so it
simply fails the magic check and is skipped. **The decoder needed no change to
its core**: `sg_ae.decode()` is called with whatever `L` survived.

One app change was required: `GBSEDApp` previously discarded files that never
completed. It now writes them (`writePartialFiles`, default true), zero-filled
where chunks are missing.

Safety relations are packed **ahead of** the structural `isIn`. An earlier
ordering led with `isIn`, which at 500 B chunks pushed `near_coll` into block 1
— where losing one chunk cost every safety relation, exactly what the format
exists to prevent. `isIn` costs actor-F1 when dropped but never costs a
braking decision.

### 10.3 Cost

| | v1 | v2 @1000 B | v2 @500 B |
|---|---|---|---|
| payload bytes (20 frames) | 16,218 | 22,000 | 20,500 |
| padding | — | 26% | 19% |
| **chunks on the wire** | **22 / 40** | **22** | **41** |

Padding inflates the bytes on disk, but **airtime is charged per chunk, and
v2 costs zero extra chunks at 1000 B and one extra at 500 B.** That is the
number that matters: the robustness is effectively free.

Constraint: a block must hold the node block plus one slice, and a slice is
`2·N²` bytes. For seq1 the minimum viable `chunkSize` is **434 B** (its
densest frame has 9 nodes). `gbsed_semantic.min_chunk_size()` computes it.

### 10.4 Result

Identical scene graphs (verified: 0 of 20 frames differ from the v1
reference), identical delivery, identical channel.

| chunkSize | config | delivered | safety recall v1 | safety recall **v2** |
|---|---|---|---|---|
| 500 | Baseline | 8/20 | 14/14 = 1.00 | 14/14 = **1.00** |
| 500 | Noise_Low | 6/20 | 8/10 = **0.80** | 10/10 = **1.00** |
| 500 | Noise_Medium | 3/20 | 2/4 = **0.50** | 4/4 = **1.00** |
| 1000 | all | — | no difference | no difference |

The single partially-delivered frame at Noise_Low tells the whole story. It
received 1 of its 2 chunks under both formats:

```
v1  frame 5  DEGRADED  safety 0/2  "payload differs from sent bytes"
v2  frame 5  DEGRADED  safety 2/2  "partial: 1 block(s) lost, 3/7 relations recovered"
```

Same bytes on the air, same chunk lost. v1 loses both safety relations; v2
keeps both and *reports precisely what is missing*.

**At 1000 B chunks the two formats are indistinguishable on this sequence**,
exactly as predicted: only 2 of 20 frames span more than one chunk, and a
single-chunk frame has nothing to reorder. The effect needs a chunk-size
regime where frames span blocks. That is a genuine limitation of the current
scenario, not of the format — denser urban scenes (slices grow as N²) would
reach it at 1000 B.

### 10.5 A side finding worth acting on

Writing partial files makes **v1 silently wrong**. A truncated v1 payload is
zero-filled, and `format_loading()` parses it happily — producing a graph that
is structurally plausible but has lost relations, with no indication. Here the
decoder still catches it by comparing SHA-256 against `meta.json`, but a real
receiver has no such reference.

v2 does not have this failure mode: each block is self-describing, so absence
is detectable. If `writePartialFiles` is left on, v2 should be the format in
use — or v1 needs a length/checksum field.

---

## 8. Metric definitions now in use

All computed once, in `gbsed_semantic.edge_metrics`, so both arms are scored by
identical code.

| metric | meaning | why |
|---|---|---|
| `delivery_rate` | frames that arrived at all | pure channel property |
| `graph_exact_rate` | reconstructions identical to ground truth | representation quality |
| `edge_f1` | F1 over all edges | comparable to prior work; **floor ≈0.38** |
| `actor_edge_f1` | F1 over edges touching a detected actor | skeleton removed; true 0–1 range |
| `risky_recall` | fraction of ego `near_coll`/`super_near` preserved | what a braking decision reads |
| `bytes_per_relation` | total bytes / safety relations preserved | efficiency, comparable across arms |
| `n_frames_classified` | frames the classifier actually saw | separates "fewer frames" from "worse frames" |
| `blocks_missing` | v2 blocks that failed the magic check | makes partial delivery visible instead of looking like a sparse scene |
| `relations_recovered` | relation slices that survived | the graceful-degradation axis |

The last one addresses a confound in the existing results: `prediction` is
sequence-level over however many frames arrived, so as delivery falls the model
sees both fewer *and different* frames. seq1 shows the symptom — `prob_class1`
goes 0.975 → 0.685 → 0.056 → 0.202, rising again at the worst channel because
one surviving frame happens to look different from five. Logging the frame
count is the minimum fix; a fixed-*k* control would be better.

---

## 9. Reproducing

```bash
# rate-semantics curve (the headline; channel-free, ~8 min)
python tools/budget_sweep.py --meta scene_data_seq1 --images images_seq1 --out budget_sweep

# matched-budget payloads
python tools/image_encode.py --meta scene_data_seq1 --images images_seq1 \
    --mode matched --codec webp --out image_data_seq1_webp
python tools/image_encode.py --meta scene_data_seq1 --images images_seq1 \
    --mode matched --codec jpeg --out image_data_seq1_jpeg
python tools/image_encode.py --meta scene_data_seq1 --images images_seq1 \
    --mode full --out image_data_seq1_full

# sweep the channel configs and score
./tools/run_image_baseline.sh image_data_seq1_webp image_data_seq1_jpeg

# combined table and figures
python tools/compare_gbsed_vs_image.py --out comparison

# slice-aligned format: encode v2 at a chunk size, then sweep v1 vs v2
python tools/gbsed_encode.py --images images_seq1 --out scene_data_seq1_v2_cs500 \
    --format v2 --chunk-size 500
./tools/run_format_sweep.sh
python tools/plot_format_sweep.py
```

| output | contents |
|---|---|
| `budget_sweep/01_rate_semantics.png` | the headline figure |
| `budget_sweep/budget_sweep.csv` | the curve's data |
| `comparison/01_gbsed_vs_image.png` | matched budget over the channel |
| `comparison/02_payload_size.png` | payload sizes, log scale |
| `comparison/comparison.csv` | per-config table |
| `experiment_results_format/01_format_sweep.png` | v1 vs v2 under partial delivery |
| `experiment_results_format/format_results.csv` | that sweep's data |
