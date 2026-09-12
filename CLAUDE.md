# GBSED — semantic encoder/decoder

This repository is the Python semantic layer: images → scene graphs →
compressed, serialized payloads, and back again.

**For results, figures and findings, read `overview.md`**, and
`comparison/ANALYSIS.md` for the full writeup.

**If you are working on the Veins/OMNeT++ simulation side, read
`../gbsed_veins/overview.md` first** — it covers the architecture, the
invariants that break silently, and machine setup. `tools/README.md` documents
the encoder and decoder themselves.

- `gbsed_semantic.py` — the shared chain. Import this, not `pipeline/pipeline.py`
  (which pulls in TensorFlow and Sionna).
- `tools/gbsed_encode.py`, `tools/gbsed_decode.py` — the CLIs.
- `Config/pipeline_extraction.yaml` — `ACTOR_NAMES` and `RELATION_NAMES` are
  index spaces that are never transmitted. Reordering either silently
  mis-decodes every previously encoded payload.

Dependencies: `numpy torch torchvision opencv-python networkx pandas
matplotlib pyyaml pydot`. Do **not** install `requirements.txt` (181 pinned
packages incl. TensorFlow/Sionna/CUDA) unless you need the original
MIMO-OFDM pipeline. detectron2 is deliberately stubbed, not installed.

`roadscene2vec` must be a sibling directory, or set `ROADSCENE2VEC_HOME`.
