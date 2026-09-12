# Results overview

Everything measured so far, what each figure shows, and which of the earlier
figures are superseded.

> Not to be confused with `../gbsed_veins/overview.md`, which orients an AI
> assistant on the *codebase*. This file is about the *results*.
> The full argument and method are in **`comparison/ANALYSIS.md`**.

---

## 1. Headline findings

**1. At a fixed bit budget, pixels recover nothing.** Re-encoding each image to
that frame's exact GBSED byte budget gives identical delivery over the same
channel — and 0/44 safety-critical relations recovered, against GBSED's 44/44.

**2. Pixels need ~96× more bandwidth to catch up.** WebP first matches GBSED's
safety-relation recall at 77,809 B/frame vs GBSED's 811. JPEG never does within
the sweep (109× and still short). As efficiency: **369 bytes per preserved
safety relation for GBSED, 35,368 for WebP.**

**3. Sending the whole frame is not merely wasteful, it is impossible here.**
24.8 MB for 20 frames — 1531× — needing 17.3 hours of simulated time against a
125 s vehicle lifetime. Measured: 1.24% of the first frame arrived.

**4. Slice-aligned packing (format v2) buys graceful degradation for free.**
At 500 B chunks, safety-relation recall under partial delivery goes from
0.80/0.50 (v1) to 1.00/1.00 (v2), with identical bytes, identical delivery, and
zero extra chunks on the wire at 1000 B.

**5. Three of the original eight channel configs were exact duplicates.**
`CAV_Good/Moderate/Bad` set `veinsmobility.x`, which TraCI overwrites. They
produced bit-identical classifier probabilities to Baseline/Noise_Medium/
Noise_High. Removed.

**6. Three of the four fidelity columns were the same number.** `bit_exact_rate
≡ delivery_rate` is real (802.11p delivers intact or not at all);
`mean_edge_f1 ≡ delivery_rate` was a bug. Fixed.

**7. Edge F1 had a free floor of ≈0.38.** The road/ego/lane skeleton is ~38% of
every edge set, so a receiver that decoded a blank image scored 0.386. Added
`actor_edge_f1`, which reads 0.000 for the same input.

---

## 2. Current figures

### 2.1 `budget_sweep/01_rate_semantics.png` — the headline

Two panels, x-axis **bytes per frame on a log scale**, channel-free.

- **Left:** fraction of safety-critical relations (`near_coll` / `super_near`
  involving ego) preserved.
- **Right:** actor-relation F1, skeleton excluded.

**How to read it.** The green star is GBSED: one point at 811 B, 1.00. The red
and purple curves are WebP and JPEG as their byte budget rises. The dotted
horizontal line is the original 1280×720 frame at unbounded budget. The green
dashed vertical line marks GBSED's budget.

**What to look for.** At GBSED's budget both curves sit at zero. They need to
travel roughly two orders of magnitude right before touching the star's height.
JPEG is pinned at exactly 0.000 through ×8 and then rises steeply — that
cliff *is* the uniform-degradation mechanism made visible.

**Why the dotted line matters.** The ground truth was produced from the
original frames, so the image arm *must* reach 1.000 at unbounded budget. It
does. That is the experiment's own sanity check: the gap is bitrate, not a
handicapped pipeline.

### 2.2 `comparison/01_gbsed_vs_image.png` — matched budget over the real channel

Three panels across the five channel conditions.

- **Left — delivery rate.** All three arms lie exactly on top of each other.
  That overlap is the point: same bytes → same chunks → same packets → same
  delivery. It proves the control held.
- **Middle — actor-relation F1 | delivered.** GBSED at 1.000, both image arms
  flat at 0.000.
- **Right — safety-critical relations preserved.** GBSED at 1.00 until the
  channel dies; image arms at 0.00 throughout.

**How to read it.** Left panel = the channel. Middle and right = the
representation. Identical inputs to the left panel, opposite outcomes in the
other two.

### 2.3 `comparison/02_payload_size.png` — payload sizes

Bar chart, log y-axis, total bytes for 20 frames, each bar labelled with its
multiple of GBSED. GBSED 16,218 B; WebP and JPEG at matched budget ≈1×
(by construction); full JPEG 24,839,399 B = 1531×.

### 2.4 `experiment_results_format/01_format_sweep.png` — v1 vs v2

Also copied to `comparison/03_format_sweep.png`.

Two panels across channel conditions, four series: v1 and v2 at 1000 B and
500 B chunks.

**How to read it.** The 1000 B pair (dashed, pale) sit on top of each other —
at that chunk size only 2 of 20 frames span more than one chunk, so there is
nothing to reorder and the formats coincide. **That overlap is expected, not a
null result.** The 500 B pair (solid) separate: red v1 falls to 0.80 then 0.50
on the left panel while green v2 holds 1.00.

**What to look for.** Left panel, Noise_Low and Noise_Medium. Same delivery
count in both arms, same chunk lost — different content recovered.

### 2.5 Per-frame scene graph renders

| directory | contents |
|---|---|
| `scene_data_seq1/png/` | the sender's graphs, 20 frames |
| `decoded_seq1/png/` | the receiver's reconstructions |

Compare `frame_XXXX_original.png` against `frame_XXXX_reconstructed.png`. For
GBSED every delivered frame is identical — same nodes (`ego car`, `car_0`,
`Left/Middle/Right Lane`, `Root Road`) and same edges.

---

## 3. Superseded figures

`result_graphs/*.png` predate the config and metric fixes. All seven plot the
eight-config set, so all carry the duplicate-config artifact. Regenerate before
using any of them.

| figure | problem |
|---|---|
| `01_delivery_rate_by_config.png` | **Sawtooth artifact.** Plots configs in list order, so `CAV_Good` jumps back to Baseline level after `Noise_High`, making it look like a recovery. It is Baseline replotted. |
| `02_bit_exact_rate_by_config.png` | **Redundant.** Bit-exact rate is identically equal to delivery rate — 802.11p never delivers a partially corrupt frame. Same curve as 01. |
| `03_fidelity_metrics_overview.png` | **Three identical panels.** Delivery rate, bit-exact rate and mean edge F1 are the same number in every row. Plus the sawtooth. The clearest illustration of both bugs — worth keeping as a "before" exhibit. |
| `04_prediction_correctness_heatmap.png` | Duplicate columns; `CAV_Extreme` missing because the sweep aborted on it. |
| `05_true_class_confidence_trend.png` | Duplicate columns. The non-monotonic tail (0.056 → 0.202) is real but confounded: it reflects the classifier seeing 1 frame instead of 5, not worse semantics. |
| `06_accuracy_summary.png` | **Biased.** Accuracy is computed over configs, so duplicated conditions are counted twice. seq1 reads 3/7 = 0.43 where the deduplicated value is 2/4 = 0.50; seq2 reads 2/7 = 0.29 against 1/4 = 0.25. |
| `07_fidelity_vs_confidence_scatter.png` | Duplicate points sit exactly on top of each other, so the scatter overstates how many independent observations there are. |

---

## 4. Data files

| file | contents |
|---|---|
| `budget_sweep/budget_sweep.csv` | the rate–semantics curve: bytes/frame, safety recall, actor F1, detections, bytes per relation |
| `comparison/comparison.csv` | matched-budget over the channel, per arm per config |
| `experiment_results_format/format_results.csv` | v1 vs v2 at both chunk sizes |
| `experiment_results/results_matrix.csv` | the original GBSED sweep (now with corrected columns) |
| `experiment_results_image/image_results_matrix.csv` | the pixel arms over the channel |
| `experiment_results*/decoded_*/fidelity.csv` | per-frame detail for every run |
| `experiment_results*/received_*/{tx,rx}_log.csv` | per-chunk logs with distance |

Per-frame columns worth knowing: `status` (EXACT / DEGRADED / CORRUPT / LOST),
`actor_edge_f1`, `risky_preserved` / `risky_orig`, `blocks_missing` and
`relations_recovered` (v2 only), `distance_m`, `chunks_heard`.

---

## 5. Regenerating

```bash
# 1. rate-semantics curve (channel-free, ~8 min)
python tools/budget_sweep.py --meta scene_data_seq1 --images images_seq1 --out budget_sweep

# 2. matched-budget payload sets
python tools/image_encode.py --meta scene_data_seq1 --images images_seq1 \
    --mode matched --codec webp --out image_data_seq1_webp
python tools/image_encode.py --meta scene_data_seq1 --images images_seq1 \
    --mode matched --codec jpeg --out image_data_seq1_jpeg
python tools/image_encode.py --meta scene_data_seq1 --images images_seq1 \
    --mode full --out image_data_seq1_full

# 3. sweep the pixel arms over the channel, then build the comparison
./tools/run_image_baseline.sh image_data_seq1_webp image_data_seq1_jpeg
python tools/compare_gbsed_vs_image.py --out comparison

# 4. slice-aligned format
python tools/gbsed_encode.py --images images_seq1 --out scene_data_seq1_v2_cs500 \
    --format v2 --chunk-size 500
./tools/run_format_sweep.sh
python tools/plot_format_sweep.py

# 5. the original GBSED sweep (paths inside need editing for this machine)
./tools/run_experiment_matrix.sh
python generate_result_graphs.py        # regenerate the superseded set
```

---

## 6. Open items

1. **No negative class.** Both sequences are labelled risky, so `match` cannot
   distinguish a working classifier from one that always predicts 1, and no
   false-positive rate is computable. Set aside by agreement; it bounds what
   the task-level numbers can claim.
2. **`result_graphs/` needs regenerating** against the deduplicated configs and
   corrected metrics.
3. **v2 is not the default.** `--format v1` still is, to stay byte-identical to
   `pipeline.GBSED._format_storage_`. With `writePartialFiles` on, a truncated
   *v1* payload parses into a silently wrong graph — prefer v2 when partial
   writing is enabled.
4. **The v2 effect needs denser scenes or smaller chunks** to show at 1000 B.
   Slices grow as N², so urban scenes would reach it naturally.
5. **Payloads are reproducible as graphs, not bytes.** Re-encoding on another
   platform gave 17/20 byte-identical payloads; the rest differ by one ULP in a
   single float16 feature. Graphs were identical in all 20.

---

## 7. Where things live

| path | what |
|---|---|
| `comparison/ANALYSIS.md` | the full writeup: design, results, why it works, every issue found |
| `gbsed_semantic.py` | the shared chain, both payload formats, all metrics |
| `tools/` | the CLIs and sweep scripts; see `tools/README.md` |
| `../gbsed_veins/overview.md` | codebase orientation, invariants, machine setup |
| `../gbsed_veins/RUNNING.md` | step-by-step run guide |
