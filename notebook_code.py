# GBSED — semantic-only round trip

```
Layer 1   image / bounding boxes   ->  roadscene2vec SceneGraph
Layer 2   SceneGraph               ->  sg_autoencoder.encode() -> sem_compression()
(Layer 3  wireless channel is deliberately NOT used here)
Layer 4   labels / features / L / comp_T  ->  sg_autoencoder.decode() -> SceneGraph'
```

Three tests:

| test | what it checks |
|---|---|
| **A** | `SceneGraph -> encode -> sem_compression -> decode -> SceneGraph'` |
| **B** | `float16 packing -> bits -> float16 -> unpacking` (no graph involved) |
| **C** | the full semantic-only round trip — everything the wireless stage would sit in the middle of |

Nothing from `Communication/` (Sionna, LDPC, OFDM, CDL, NeuralReceiver) is imported.
This is the notebook form of `load_model.py`; run the cells top to bottom.
## 1. Paths

`roadscene2vec` is not pip-installed — it lives next to this repo.
import sys
import types
import argparse
from pathlib import Path

import numpy as np

# In a notebook __file__ is not defined, so fall back to the current working directory.
try:
    GBSED_ROOT = Path(__file__).resolve().parent
except NameError:
    GBSED_ROOT = Path.cwd()

if not (GBSED_ROOT / "sgautoencoder").is_dir():
    GBSED_ROOT = Path("E:/Project/Project/gbsed")


def _find_roadscene2vec():
    """Return the directory that must be on sys.path for `import roadscene2vec` to work."""
    candidates = [
        GBSED_ROOT.parent / "roadscene2vec",          # sibling checkout of the repo
        GBSED_ROOT / "roadscene2vec",                 # vendored inside this repo
    ]
    for root in candidates:
        if (root / "roadscene2vec" / "__init__.py").is_file():
            return root
    try:                                              # already installed in the env
        import roadscene2vec  # noqa: F401
        return None
    except ImportError:
        raise SystemExit(
            "Could not locate roadscene2vec. Expected one of:\n  "
            + "\n  ".join(str(c) for c in candidates)
        )


_R2V_ROOT = _find_roadscene2vec()
if _R2V_ROOT is not None and str(_R2V_ROOT) not in sys.path:
    sys.path.insert(0, str(_R2V_ROOT))
if str(GBSED_ROOT) not in sys.path:                   # so `sgautoencoder.*` imports
    sys.path.insert(0, str(GBSED_ROOT))

print("gbsed        :", GBSED_ROOT)
print("roadscene2vec:", _R2V_ROOT)
## 2. detectron2 stub

`roadscene2vec/scene_graph/extraction/image_extractor.py` imports detectron2 at module level,
and `sgautoencoder/sg_autoencoder.py` imports that module. detectron2 is not installed here
(and its `DefaultPredictor` defaults to `cfg.MODEL.DEVICE == "cuda"`, so it would need an
NVIDIA GPU anyway). We only need `RealExtractor`'s **relation_extractor** and **bev**
attributes for encode/decode, so we stub the import out and replace the class.
def _stub_detectron2():
    try:
        import detectron2
        # re-running this cell must not mistake our own stub for the real thing
        return getattr(detectron2, "_gbsed_stub", False)
    except ImportError:
        pass

    def _module(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    d2 = _module("detectron2")
    engine = _module("detectron2.engine")
    data = _module("detectron2.data")
    utils = _module("detectron2.utils")
    vis = _module("detectron2.utils.visualizer")
    config = _module("detectron2.config")
    zoo = _module("detectron2.model_zoo")

    def _unavailable(*_a, **_kw):
        raise RuntimeError(
            "detectron2 is not installed; this code path is stubbed out. "
            "Scene graphs here are built from bounding boxes supplied directly."
        )

    engine.DefaultPredictor = _unavailable
    data.MetadataCatalog = _unavailable
    config.get_cfg = _unavailable
    zoo.get_config_file = _unavailable
    zoo.get_checkpoint_url = _unavailable
    vis.Visualizer = _unavailable

    utils.visualizer = vis
    d2.engine, d2.data, d2.utils, d2.config, d2.model_zoo = engine, data, utils, config, zoo
    d2._gbsed_stub = True
    return True


_D2_STUBBED = _stub_detectron2()
print("detectron2 stubbed:", _D2_STUBBED)
## 3. roadscene2vec imports, `LiteExtractor`, and the `sg_autoencoder`
from roadscene2vec.util.config_parser import configuration
from roadscene2vec.scene_graph.scene_graph import SceneGraph
from roadscene2vec.scene_graph.extraction import extractor as base_ex
from roadscene2vec.scene_graph.extraction import image_extractor
from roadscene2vec.scene_graph.extraction.bev import bev as bev_mod


class LiteExtractor(base_ex.Extractor):
    """
    Everything RealExtractor provides that the semantic layer actually touches:
    a RelationExtractor and a calibrated BEV. No detectron2, no dataset directory.
    """

    def __init__(self, config):
        super(LiteExtractor, self).__init__(config)
        self.bev = bev_mod.BEV(config.image_settings["BEV_PATH"], mode="deploy")


# sg_autoencoder.__init__ does `RealEx.RealExtractor(self.config)`; the lookup happens at
# call time, so patching the module attribute before the import below is enough.
image_extractor.RealExtractor = LiteExtractor

from sgautoencoder.sg_autoencoder import sg_autoencoder

print("sg_autoencoder imported, RealExtractor ->", image_extractor.RealExtractor.__name__)
## 4. Config

`Config/pipeline_extraction.yaml` ships with relative paths that only resolve from `pipeline/`;
we load it and rewrite the paths we need to absolute ones.
def load_config(yaml_path=None):
    yaml_path = Path(yaml_path or GBSED_ROOT / "Config" / "pipeline_extraction.yaml")
    cfg = configuration(str(yaml_path), from_function=True)

    bev_path = Path(cfg.image_settings["BEV_PATH"])
    if not bev_path.is_file():
        pkg = Path(image_extractor.__file__).resolve().parent      # .../scene_graph/extraction
        bev_path = pkg / "bev" / "bev.json"
    cfg.image_settings["BEV_PATH"] = str(bev_path)
    return cfg
## 5. Serialization helpers

Copied verbatim from `pipeline/pipeline.py` (`GBSED._format_storage_`, `_format_loading_`,
`_to_bits_array_`, `_to_float_array_`). Importing `pipeline.py` would drag in Sionna and the
whole MIMO-OFDM stack, which is exactly what this experiment excludes.
def format_storage(labels, feature_nodes, L, comp_T):
    to_serialize = []
    to_serialize.append(len(labels))
    to_serialize.extend(labels)
    to_serialize.append(feature_nodes.size)
    to_serialize.extend(feature_nodes.ravel())
    to_serialize.append(L.size)
    to_serialize.extend(L)
    to_serialize.append(comp_T.size)
    to_serialize.extend(comp_T.ravel())
    to_serialize = np.asarray(to_serialize, dtype=np.float16)
    return to_serialize


def format_loading(to_read):
    cur_idx, end_idx, nb = 0, 0, 0

    nb = int(to_read[cur_idx]); cur_idx += 1
    end_idx = cur_idx + nb
    labels = to_read[cur_idx:end_idx]

    cur_idx = end_idx
    nb = int(to_read[cur_idx]); cur_idx += 1
    end_idx = cur_idx + nb
    features = to_read[cur_idx:end_idx]

    cur_idx = end_idx
    nb = int(to_read[cur_idx]); cur_idx += 1
    end_idx = cur_idx + nb
    L = to_read[cur_idx:end_idx]

    cur_idx = end_idx
    nb = int(to_read[cur_idx]); cur_idx += 1
    end_idx = cur_idx + nb
    comp_T = to_read[cur_idx:]

    labels = [int(i) for i in labels]
    feature_nodes = features.reshape(((len(labels)), -1))
    L = [int(i) for i in L]
    comp_T = comp_T.reshape((len(L), len(labels), len(labels)))

    return labels, feature_nodes, L, comp_T


def to_bits_array(np_array):
    b = np_array.tobytes()
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def to_float_array(bits):
    b = np.packbits(bits)
    return np.frombuffer(b.tobytes(), np.float16)
## 6. Layer 1 — getting a `SceneGraph`

`detect_boxes()` stands in for detectron2's Mask R-CNN, which is not installed and whose
`DefaultPredictor` defaults to CUDA. torchvision's Faster R-CNN gives the same
`(boxes, labels, image_size)` triple and runs on CPU or MPS. It downloads ~160 MB of COCO
weights on first call.
# torchvision's detection models index into the 91-entry COCO list (with "N/A" holes).
COCO_CLASS_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


def scene_graph_from_boxes(cfg, bev, boxes, labels, image_size, class_names=COCO_CLASS_NAMES):
    """Build a real roadscene2vec SceneGraph from detector output."""
    import roadscene2vec.scene_graph.relation_extractor as r_e
    relation_extractor = r_e.RelationExtractor(cfg)
    return SceneGraph(
        relation_extractor,
        bounding_boxes=(boxes, labels, image_size),
        bev=bev,
        coco_class_names=class_names,
        platform=cfg.dataset_type,
    )


def demo_boxes():
    """
    Three cars in a 1280x720 frame -- the resolution bev.json is calibrated for.
    Boxes are [left, top, right, bottom]; label 3 == 'car' in the COCO list above.
    """
    import torch
    boxes = torch.tensor([
        [560.0, 380.0, 700.0, 480.0],    # ahead, same lane
        [300.0, 400.0, 460.0, 520.0],    # ahead-left
        [900.0, 390.0, 1050.0, 500.0],   # ahead-right
    ])
    return boxes, [3, 3, 3], (720, 1280)


def detect_boxes(image_path, device="cpu", score_thresh=0.5):
    """
    Stand-in for detectron2's Mask R-CNN, which is not installed and whose
    DefaultPredictor defaults to CUDA.

    Downloads ~160 MB of COCO weights on first call.
    """
    import cv2
    import torch
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)

    im = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit("Could not read image: %s" % image_path)
    im = cv2.resize(im, (1280, 720))                 # match the BEV calibration
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval().to(device)
    with torch.no_grad():
        out = model([tensor.to(device)])[0]

    keep = out["scores"] >= score_thresh
    boxes = out["boxes"][keep].cpu()
    labels = [int(l) for l in out["labels"][keep].cpu()]
    return boxes, labels, (720, 1280)
## 7. Inspection / comparison helpers
def describe_graph(sg, title):
    nodes = list(sg.g.nodes)
    print("\n%s" % title)
    print("  nodes (%d): %s" % (len(nodes), [n.name for n in nodes]))
    print("  edges (%d)" % sg.g.number_of_edges())


def edge_set(sg):
    return sorted(
        (src.name, data["label"], dst.name) for src, dst, data in sg.g.edges(data=True)
    )


def compare_graphs(sg, rec_sg):
    """Return (ok, report_lines) comparing node names/labels and the relation multiset."""
    lines, ok = [], True

    orig_nodes = [(n.name, n.label, n.value) for n in sg.g.nodes]
    rec_nodes = [(n.name, n.label, n.value) for n in rec_sg.g.nodes]
    if orig_nodes == rec_nodes:
        lines.append("  nodes            : identical (%d)" % len(orig_nodes))
    else:
        ok = False
        lines.append("  nodes            : DIFFER")
        lines.append("    original     : %s" % orig_nodes)
        lines.append("    reconstructed: %s" % rec_nodes)

    e0, e1 = edge_set(sg), edge_set(rec_sg)
    if e0 == e1:
        lines.append("  relations        : identical (%d)" % len(e0))
    else:
        ok = False
        only_orig = [e for e in e0 if e not in e1]
        only_rec = [e for e in e1 if e not in e0]
        lines.append("  relations        : DIFFER (%d vs %d)" % (len(e0), len(e1)))
        lines.append("    missing in rec : %s" % only_orig[:10])
        lines.append("    extra in rec   : %s" % only_rec[:10])
    return ok, lines


def maybe_visualize(sg, path):
    try:
        sg.visualize(str(path))
        print("  wrote %s" % path)
    except Exception as e:                                    # graphviz `dot` missing
        print("  could not render %s (%s: %s)" % (path, type(e).__name__, e))
## 8. The three tests
def test_a(ae, sg):
    """SceneGraph -> encode -> sem_compression -> decode -> SceneGraph'."""
    print("\n" + "=" * 78)
    print("TEST A -- pure semantic encode / decode (no serialization)")
    print("=" * 78)

    labels, feat_nodes_mat, T = ae.encode(sg)
    comp_T, L = ae.sem_compression(T)

    print("  number of nodes  : %d" % len(labels))
    print("  labels           : %s" % labels)
    print("  actor names      : %s"
          % [ae.config.relation_extraction_settings["ACTOR_NAMES"][i] for i in labels])
    print("  feature matrix   : shape=%s dtype=%s" % (feat_nodes_mat.shape, feat_nodes_mat.dtype))
    print("  relation tensor T: shape=%s dtype=%s  (nonzero=%d)"
          % (T.shape, T.dtype, int(np.count_nonzero(T))))
    print("  compressed comp_T: shape=%s  (%.1f%% of T)"
          % (comp_T.shape, 100.0 * comp_T.size / max(T.size, 1)))
    print("  active relations : L=%s -> %s"
          % (list(L), [ae.rels[i] for i in L]))

    # sem_decompression() does `L.index(i)`, so it needs a Python list, not the
    # np.int8 array sem_compression() hands back.  In the wireless path this
    # conversion happens inside _format_loading_(); here we do it explicitly.
    rec_sg = ae.decode(labels, feat_nodes_mat, list(L), comp_T)

    describe_graph(sg, "original SceneGraph")
    describe_graph(rec_sg, "reconstructed SceneGraph")

    ok, lines = compare_graphs(sg, rec_sg)
    print("\n  comparison:")
    for line in lines:
        print(line)
    print("\n  TEST A: %s" % ("PASS" if ok else "FAIL"))
    return ok, rec_sg, (labels, feat_nodes_mat, comp_T, L)


def test_b(labels, feat_nodes_mat, comp_T, L):
    """float16 packing -> bits -> float16 -> unpacking."""
    print("\n" + "=" * 78)
    print("TEST B -- binary serialization round trip (no graph)")
    print("=" * 78)

    packed = format_storage(labels, feat_nodes_mat, L, comp_T)
    bits = to_bits_array(packed)
    print("  packed float16   : %d values (%d bytes)" % (packed.size, packed.nbytes))
    print("  bit array        : %d bits" % bits.size)

    unpacked = to_float_array(bits)
    rec_labels, rec_feat, rec_L, rec_comp_T = format_loading(unpacked)

    checks = [
        ("labels        ", np.array_equal(np.asarray(labels), np.asarray(rec_labels))),
        ("feature matrix", np.array_equal(feat_nodes_mat, rec_feat)),
        ("L             ", np.array_equal(np.asarray(L), np.asarray(rec_L))),
        ("comp_T        ", np.array_equal(comp_T.astype(np.float16), rec_comp_T)),
    ]
    for name, good in checks:
        print("  %s : %s" % (name, "exact" if good else "MISMATCH"))

    ok = all(good for _, good in checks)
    print("\n  TEST B: %s" % ("PASS" if ok else "FAIL"))
    return ok


def test_c(ae, sg):
    """The full semantic-only chain -- exactly Test C from the handoff."""
    print("\n" + "=" * 78)
    print("TEST C -- full semantic-only round trip (Layer 1 -> 2 -> 4, no channel)")
    print("=" * 78)

    labels, feat_nodes_mat, T = ae.encode(sg)
    comp_T, L = ae.sem_compression(T)

    packed = format_storage(labels, feat_nodes_mat, L, comp_T)
    bits = to_bits_array(packed)

    # This is where MIMOE2EModel would sit.  Here the "channel" is the identity.
    received_bits = np.asarray(bits, dtype=np.uint8)

    rec_labels, rec_feat, rec_L, rec_comp_T = format_loading(to_float_array(received_bits))
    rec_sg = ae.decode(rec_labels, rec_feat, rec_L, rec_comp_T)

    print("  transmitted      : %d bits (%.2f kB)" % (bits.size, bits.size / 8 / 1024))
    print("  payload          : %d nodes, %d active relations" % (len(labels), len(L)))

    ok, lines = compare_graphs(sg, rec_sg)
    print("\n  comparison:")
    for line in lines:
        print(line)
    print("\n  TEST C: %s" % ("PASS" if ok else "FAIL"))
    return ok, rec_sg
## 9. Run it

Set `IMAGE_PATH` to a driving image to run the real detector, or leave it `None` to use the
synthetic bounding boxes (no downloads). `DEVICE` is `"cpu"` or `"mps"` — there is no CUDA
on this machine.
IMAGE_PATH = "images/00097100.png"      # e.g. "images/road.jpg"
# IMAGE_PATH = None
DEVICE = "cpu"             # "cpu" or "mps"
SCORE_THRESH = 0.5

if _D2_STUBBED:
    print("note: detectron2 not installed -- using the stub + torchvision detector")

cfg = load_config()
print("config           : %s" % cfg.yaml_path)
print("bev calibration  : %s" % cfg.image_settings["BEV_PATH"])
print("relations        : %s" % cfg.relation_extraction_settings["RELATION_NAMES"])

ae = sg_autoencoder(cfg)
bev = ae.sg_extraction_object.bev
### Layer 1 — build the SceneGraph
if IMAGE_PATH:
    print("detector         : torchvision fasterrcnn_resnet50_fpn on %s" % DEVICE)
    boxes, labels_, image_size = detect_boxes(IMAGE_PATH, DEVICE, SCORE_THRESH)
    print("detections       : %d above %.2f -> %s"
          % (len(labels_), SCORE_THRESH, [COCO_CLASS_NAMES[i] for i in labels_]))
else:
    print("detector         : none (synthetic bounding boxes)")
    boxes, labels_, image_size = demo_boxes()

sg = scene_graph_from_boxes(cfg, bev, boxes, labels_, image_size)

if sg.g.number_of_nodes() <= 5:
    print("\nwarning: no traffic participants made it into the graph -- only the "
          "road/ego/lane skeleton is present. Check that the detector found actors "
          "whose COCO names appear in ACTOR_NAMES / *_NAMES in the yaml.")

describe_graph(sg, "extracted SceneGraph")
### Test A — pure semantic encode / decode
ok_a, rec_a, (labels, feats, comp_T, L) = test_a(ae, sg)
### Test B — binary serialization round trip
ok_b = test_b(labels, feats, comp_T, L)
### Test C — full semantic-only round trip
ok_c, rec_c = test_c(ae, sg)
### Summary
print("=" * 78)
print("SUMMARY   A=%s   B=%s   C=%s"
      % tuple("PASS" if o else "FAIL" for o in (ok_a, ok_b, ok_c)))
print("=" * 78)
### 10. Generate One GBSED Payload
This creates the binary payload and saves it for Veins.
from pathlib import Path

PAYLOAD_DIR = Path("scene_data")
PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)

print("Payload directory:", PAYLOAD_DIR.resolve())
# Encode the SceneGraph
labels, feature_nodes, T = ae.encode(sg)

# Semantic compression
comp_T, L = ae.sem_compression(T)

# Serialize the representation
packed = format_storage(
    labels,
    feature_nodes,
    L,
    comp_T
)

print("GBSED payload created")
print("----------------------")
print("Labels              :", len(labels))
print("Feature matrix      :", feature_nodes.shape)
print("Active relations    :", len(L))
print("Packed values       :", packed.size)
print("Payload size        :", packed.nbytes, "bytes")
print("Payload dtype       :", packed.dtype)
payload_bytes = packed.tobytes()

print("Number of bytes:", len(payload_bytes))

assert len(payload_bytes) == packed.nbytes
print("Byte conversion: PASS")
payload_path = PAYLOAD_DIR / "scene_0000.bin"

with open(payload_path, "wb") as f:
    f.write(payload_bytes)

print("Saved:", payload_path.resolve())
print("Size :", payload_path.stat().st_size, "bytes")
import numpy as np

with open(payload_path, "rb") as f:
    received_bytes = f.read()

received_float16 = np.frombuffer(
    received_bytes,
    dtype=np.float16
)

rec_labels, rec_features, rec_L, rec_comp_T = format_loading(
    received_float16
)

rec_sg_file = ae.decode(
    rec_labels,
    rec_features,
    rec_L,
    rec_comp_T
)

ok, lines = compare_graphs(sg, rec_sg_file)

for line in lines:
    print(line)

print("\nFILE ROUND TRIP:", "PASS" if ok else "FAIL")
with open(payload_path, "rb") as f:
    file_bytes = f.read()

assert file_bytes == payload_bytes

print("Binary byte integrity: PASS")
print("Payload size:", len(file_bytes), "bytes")
import json

metadata = {
    "scene_id": 0,
    "payload_file": "scene_0000.bin",
    "payload_bytes": len(payload_bytes),
    "num_nodes": len(labels),
    "num_active_relations": len(L),
}

metadata_path = PAYLOAD_DIR / "scene_0000.json"

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved:", metadata_path.resolve())
### Optional — render the graphs

Needs `pydot` and the graphviz `dot` binary; it reports and moves on if they are missing.
VISUALIZE_DIR = "out"    # e.g. "out"

if VISUALIZE_DIR:
    outdir = Path(VISUALIZE_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    maybe_visualize(sg, outdir / "original.png")
    maybe_visualize(rec_c, outdir / "reconstructed.png")
---

## Appendix — the original `.h5` inspection

Kept from the earlier session. `Communication/weights/neural_rx_ofdm_mimo_cdl_final.h5` is
not HDF5 at all: it is a pickle holding 263 TensorFlow `ResourceVariable`s for the
`NeuralReceiver`. Nothing above depends on these cells.