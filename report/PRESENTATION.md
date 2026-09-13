# Presentation content — 10 minutes, 8 slides

Structured to the official IUT slide-by-slide guidelines, covering every item
the course brief lists. Figure paths are relative to `report/figures/`.

**Timing plan (strict 10 min):** Slides 1–3 ≈ 3 min · Slides 4–5 ≈ 2.5 min ·
Slide 6 ≈ 3 min · Slides 7–8 ≈ 1.5 min. The results slide is the one worth
protecting — cut from 4 and 5 if you are running long, never from 6.

**Speaker split:** one member per block (1–3, 4–5, 6–8) keeps the "balanced
speaking" criterion satisfied. Rehearse the handoffs; they cost 15 seconds each
if unplanned.

---

## Slide 1 — Title

**Semantic Communication for Connected Autonomous Vehicles:**
**Evaluating a Graph-Based Encoder–Decoder over a Realistic VANET**

- **Course:** CSE 4610 / SWE 4606 — Design Project
- **Team:**
  - `<<Name>>` — `<<ID>>` — Veins application layer & scenario design
  - `<<Name>>` — `<<ID>>` — Semantic tooling & experiment harness
  - `<<Name>>` — `<<ID>>` — Pixel baseline & evaluation metrics
- **Supervisor:** `<<Name>>`, `<<Designation>>` — `<<Research Lab / Group>>`
- **Affiliation:** Department of CSE, Islamic University of Technology

> *Say:* "We took a published semantic communication system and asked one
> question its own evaluation could not answer — at the same number of bytes,
> is a scene graph actually better than an image?"

---

## Slide 2 — Problem & Motivation

**Real-world context**
- Connected vehicles need to share what they see — cooperative perception saves
  lives when one car sees what another cannot.
- One camera frame ≈ **1.2 MB**. IEEE 802.11p offers ~6 Mbit/s *shared*, and
  two passing vehicles stay in range for only tens of seconds.

**The measured consequence** *(our result, not a citation)*
- Transmitting 20 raw frames needs **17.3 hours** of channel time.
- The vehicles are in range for **125 seconds**.
- We ran it: **1.24 % of a single frame arrived.**

**Problem statement**
> Transmit the *meaning* of a road scene instead of its pixels — and prove the
> advantage under experimental control, not against a strawman baseline.

**Broader impact**
- Spectrum is shared and finite — bandwidth taken is bandwidth denied to others.
- Scene graphs contain no faces and no licence plates — **privacy by
  construction**.
- Low bandwidth ⇒ cheap radios ⇒ safety benefits are not restricted to premium
  vehicles.

---

## Slide 3 — Related Work & Gap

**What exists**

| Work | Contribution |
|---|---|
| `roadscene2vec` (Malawade et al., 2022) | Extracts road scene graphs; CNNs can't capture inter-object relations |
| Scene-Graph Risk Assessment (Yu et al., 2021) | Graphs beat pixels for risk: **96.4 % vs 91.2 %**, and transfer better (87.8 % vs 70.3 %) |
| **GBSED** (Ribouh & Di Ngoma, 2026) | Scene graph → semantic compression → MIMO–OFDM channel. Reports **99.9 %** compression, fidelity > 0.9 above 10 dB |

**The two gaps we target**

1. **The channel is not a network.** GBSED is evaluated on an abstract SNR
   sweep — no mobility, no MAC, no packet loss, no vehicle leaving range.
2. **The baseline is too weak.** "99.9 % smaller than *raw RGB*" — but nobody
   transmits raw pixels. What does a real codec do at the *same byte budget*?

**Our proposal**
> Port GBSED onto OMNeT++ / Veins / SUMO over 802.11p, and build a **controlled
> fixed-budget comparison** where identical packets cross an identical channel
> and only the *meaning* of the bytes differs.

---

## Slide 4 — System Architecture

**Figure:** `architecture.png` *(full slide — it is the anchor of the talk)*

**The pipeline**
`image → detector → BEV projection → scene graph → encode(T, F) → semantic compression → serialise → [802.11p] → deserialise → decompress → graph′ → MR-GCN + LSTM risk classifier`

**Key design decision — the semantic layer never runs inside the simulation**
- Encoding is a pre-process, decoding a post-process; the network moves opaque
  bytes.
- Justified: the encoder's output depends on nothing the simulation knows.
- In-loop would cost ~1 s/frame on a run that takes 0.3 s — **65× slower for
  zero benefit.**

**Design trade-off worth naming**
- The shared codebook (actor and relation name lists) is **never transmitted**
  — node names are *re-derived* at the receiver from label indices.
- Saves bytes; creates a hard coupling. Reorder either list and every previously
  encoded payload silently decodes wrong.

---

## Slide 5 — Implementation

**Stack:** OMNeT++ 6.4.0 · Veins 5.3.1 · SUMO 1.21.0 · IEEE 802.11p (6 Mbit/s,
20 mW) · PyTorch 2.8 · `roadscene2vec` · C++17 + Python 3.12

**What we built on the network side**
- `GBSEDApp` (C++): chunking, Base64, reassembly over 802.11p
- Added **configurable send timing**, **sender position stamping**, **per-chunk
  CSV logging** (this is what makes delivery analysable against distance), and
  **partial-file delivery**
- Reworked the SUMO scenario: vehicles now **diverge at right angles at 8 m/s**

> *Why that matters:* originally both vehicles drove the same road at the same
> speed, so nothing was ever lost — delivery was 100 % and the simulation
> measured nothing. 8 m/s was chosen so both vehicles **outlive the send
> queue**, isolating distance as the only variable.

**What we built on the semantic side**
- Consolidated duplicated serialisation code into one module
- Encoder / decoder CLIs, channel sweep harness, risk-assessment integration
- **Pixel baseline**: re-encode each image to that frame's *exact* byte budget
- **Format v2**: slice-aligned payload (Slide 6)

**Figure:** `scenario.png` — geometry + measured range cliff at **488 m**
(separation grows at 10.35 m/s, r = 0.99999 against the app's own logs)

---

## Slide 6 — Results *(the core slide — budget ~3 min)*

### 6a. The chain survives a real network
Every delivered frame reconstructs **node- and edge-identically**.
Actor-edge F1 = **1.000**, safety-relation recall = **1.000**, in every config.
*In 802.11p, degradation is about* which *frames arrive, not how damaged they are.*

**Figure:** `sg_original.png` + `sg_reconstructed.png` side by side

### 6b. Fixed budget: same bytes, same channel
**Figure:** `matched_budget.png`

| Arm | Delivered | Graph exact | Safety relations |
|---|---|---|---|
| **GBSED** | 70 % | **70 %** | **32/32** |
| WebP @ same bytes | 70 % | **0 %** | **0/32** |
| JPEG @ same bytes | 70 % | **0 %** | **0/32** |

> *Say:* "The three overlapping lines in the left panel are the point — the
> control held. Same packets, same radio. At that budget an image is 102×58
> pixels at quality 1. The detector found **zero** objects in the JPEGs."

### 6c. How much more bandwidth would pixels need?
**Figure:** `rate_semantics.png`

- WebP first matches GBSED's safety recall at **77,809 B/frame — 96×**
- JPEG never reaches it within 128×
- Efficiency: **369 bytes per safety relation** vs WebP's **35,368**
- **Validity control:** the dotted line is the original image at unbounded
  budget. The ground truth came *from* those images, so it **must** reach 1.000
  — and it does. The gap is bitrate, not a rigged pipeline.

### 6d. Graceful degradation (our extension)
**Figure:** `format_sweep.png`

Semantic decompression *already* tolerated a partial relation set — only the
packing threw it away. Format v2 aligns chunk boundaries to relation slices,
safety-first.

| | v1 | **v2** |
|---|---|---|
| Safety recall, Noise_Medium | 0.50 | **1.00** |
| Extra chunks on the wire | — | **0** |

**Figure:** `partial_delivery.png` — one frame, one lost chunk, both layouts

> *Say:* "Same frame, same chunk lost. Both reconstructions have **exactly ten
> edges** and an **identical** actor-F1 of 0.667 — but v2 keeps both safety
> relations and v1 keeps neither. An aggregate structural metric cannot tell
> these two apart. That is why we report safety-relation recall separately."

---

## Slide 7 — Challenges, Limitations & Future Work

**Challenges**
- Toolchain: SUMO version constrained from three directions at once (Veins
  accepts TraCI API 15–21 only); upstream build-blocking compile error
- **Silent failures** were the real cost — each produced plausible *wrong*
  answers rather than errors

**Evaluation defects we found and fixed** *(worth 30 seconds — shows rigour)*
1. Three of four fidelity metrics were **the same number**. Cause: the filter
   `if r["edge_f1"]` — lost frames store the *string* `"0.0"`, which is truthy.
2. Edge F1 had a **free floor of 0.38** — the road/lane skeleton. A receiver
   decoding a *blank image* scored 0.386. Our actor-edge F1 reads 0.000.
3. Three of eight channel configs were **exact duplicates** — they set a
   mobility parameter TraCI overwrites. Proof: bit-identical classifier
   probabilities (0.9753320813179016).

**Limitations (state them plainly)**
- No negative class — accuracy claims are bounded
- Two sequences, two nodes; no MAC contention
- The v2 benefit needs small chunks or denser scenes to appear

**Future work**
- **Adaptive encoding** — send only safety slices when the link is poor; needs
  no ML in the loop
- Multi-vehicle contention · balanced dataset · close the loop back into SUMO

---

## Slide 8 — Conclusion & Q/A

**What we showed**
1. The semantic chain is **lossless** over a realistic VANET
2. At an identical budget: **44/44 vs 0/44** safety relations
3. Images need **~96× more bandwidth** for the same safety content
4. Whole frames are **infeasible**, not merely expensive (1.24 % of one frame)
5. Slice-aligned packing: partial loss costs a *relation*, not a *frame*

**The honest framing** *(say this — it pre-empts the hardest question)*
> GBSED is not better *compression*. It transmits the **output** of perception
> rather than its **input**, and wins because the receiver's task needs only
> that output. The corollary: the receiver can never do anything the sender's
> detector did not already encode. A raw image is task-agnostic; a scene graph
> is committed.

**GitHub**
- `https://github.com/sahil0319/gbsed` — semantic layer, tooling, experiments
- `https://github.com/Loona6/gbsed_veins` — Veins application & scenario

**Individual contributions**

| Member | Owned |
|---|---|
| `<<Name>>` | Veins application layer, CSV logging, scenario redesign |
| `<<Name>>` | Semantic module consolidation, encoder/decoder CLIs, channel sweep |
| `<<Name>>` | Pixel baseline, rate–semantics sweep, metric design & defect analysis |

**Thank you — questions?**

---

## Anticipated Q&A

**"Isn't comparing against a 102×58 image unfair?"**
That *is* the comparison — it's what the byte budget buys. We also swept the
budget upward until images caught up, which is the 96× number. And we gave WebP
every advantage: it beats JPEG at low bitrate, came in *under* budget, and held
256×144. It still recovered nothing.

**"How do you know your image pipeline isn't just broken?"**
Built-in control: at unbounded budget the image arm scores exactly 1.000,
because the ground truth was generated from those same images. If the pipeline
were broken it could not reach 1.000.

**"Why 802.11p rather than the paper's MIMO–OFDM?"**
Complementary, not a replacement. Theirs is a physical-layer study; ours adds
mobility, MAC and packet loss. The finding that only 802.11p can show is that
loss is *bursty and distance-correlated* — fidelity is perfect or the frame is
absent, never partially degraded.

**"What happens if the detector misses an object?"**
It is lost permanently — the receiver has no pixels to re-examine. This is the
fundamental trade of semantic communication and we state it as a limitation.
Deployments should keep raw data locally for forensics even while transmitting
only semantics.

**"Why is delivery identical across all three arms?"**
By construction — same bytes ⇒ same chunk count ⇒ same packets. That overlap is
the experimental control, and verifying it is what licenses every other claim
on the slide.
