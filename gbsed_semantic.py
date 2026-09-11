#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The GBSED semantic layer, importable.

Everything here was previously duplicated between `load_model.ipynb` (cells 2-16)
and `pipeline/pipeline.py`.  It is the encode/serialize/deserialize/decode chain
with the wireless stage removed, so a transport other than Sionna's MIMO-OFDM
model can sit in the middle -- in our case Veins/OMNeT++ over IEEE 802.11p.

    Layer 1   image             ->  SceneGraph                (roadscene2vec)
    Layer 2   SceneGraph        ->  encode() -> sem_compression()
    Layer 3   arrays            ->  format_storage() -> bits  ->  .bin
              << transport >>
    Layer 4   .bin -> bits      ->  format_loading() -> decode() -> SceneGraph'

Importing this module has side effects, and their order matters:
detectron2 is stubbed out before roadscene2vec imports it, and
`image_extractor.RealExtractor` is replaced by `LiteExtractor` before
`sgautoencoder` looks it up.
"""

import os
import sys
import types
import hashlib
from pathlib import Path

import numpy as np

__all__ = [
    "GBSED_ROOT", "COCO_CLASS_NAMES",
    "load_config", "make_autoencoder",
    "format_storage", "format_loading", "to_bits_array", "to_float_array",
    "pack_to_bytes", "unpack_from_bytes",
    "detect_boxes", "demo_boxes", "scene_graph_from_boxes",
    "encode_scene_graph", "decode_payload",
    "graph_summary", "compare_graphs", "compare_summary", "edge_set",
    "describe_graph", "maybe_visualize", "sha256_of",
]


# --------------------------------------------------------------------------
# 1. Paths -- roadscene2vec is not pip-installed, it lives next to this repo
# --------------------------------------------------------------------------

GBSED_ROOT = Path(__file__).resolve().parent


def _find_roadscene2vec():
    """Return the directory that must be on sys.path for `import roadscene2vec`.

    Search order: the ROADSCENE2VEC_HOME environment variable, a sibling
    checkout, a vendored copy, then whatever is already importable. Nothing
    here is machine-specific, so a fresh clone works as long as roadscene2vec
    sits next to this repo (the layout its own README assumes).
    """
    candidates = []
    env = os.environ.get("ROADSCENE2VEC_HOME")
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [
        GBSED_ROOT.parent / "roadscene2vec",   # sibling checkout
        GBSED_ROOT / "roadscene2vec",          # vendored
    ]
    for root in candidates:
        if (root / "roadscene2vec" / "__init__.py").is_file():
            return root
        # tolerate ROADSCENE2VEC_HOME pointing at the inner package directory
        if root.name == "roadscene2vec" and (root / "__init__.py").is_file():
            return root.parent
    try:
        import roadscene2vec  # noqa: F401
        return None                            # already importable
    except ImportError:
        raise SystemExit(
            "Could not locate roadscene2vec. Looked in:\n  "
            + "\n  ".join(str(c) for c in candidates)
            + "\n\nClone it next to this repo:\n"
              "  git clone https://github.com/AICPS/roadscene2vec %s\n"
              "or set ROADSCENE2VEC_HOME to wherever it already lives."
              % (GBSED_ROOT.parent / "roadscene2vec")
        )


_R2V_ROOT = _find_roadscene2vec()
if _R2V_ROOT is not None and str(_R2V_ROOT) not in sys.path:
    sys.path.insert(0, str(_R2V_ROOT))
if str(GBSED_ROOT) not in sys.path:
    sys.path.insert(0, str(GBSED_ROOT))


# --------------------------------------------------------------------------
# 2. detectron2 stub
# --------------------------------------------------------------------------
# `image_extractor.py` imports detectron2 at module level and `sg_autoencoder`
# imports that module.  detectron2 is not installed here, and its
# DefaultPredictor defaults to cfg.MODEL.DEVICE == "cuda" anyway.  Only
# RealExtractor's `relation_extractor` and `bev` attributes are ever touched by
# the semantic layer, so the import is stubbed and the class replaced below.

def _stub_detectron2():
    try:
        import detectron2
        return getattr(detectron2, "_gbsed_stub", False)   # don't re-stub our own stub
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


D2_STUBBED = _stub_detectron2()


# --------------------------------------------------------------------------
# 3. roadscene2vec imports, LiteExtractor, sg_autoencoder
# --------------------------------------------------------------------------

from roadscene2vec.util.config_parser import configuration          # noqa: E402
from roadscene2vec.scene_graph.scene_graph import SceneGraph        # noqa: E402
from roadscene2vec.scene_graph.extraction import extractor as base_ex        # noqa: E402
from roadscene2vec.scene_graph.extraction import image_extractor              # noqa: E402
from roadscene2vec.scene_graph.extraction.bev import bev as bev_mod           # noqa: E402


class LiteExtractor(base_ex.Extractor):
    """
    Everything RealExtractor provides that the semantic layer actually touches:
    a RelationExtractor (from the base class) and a calibrated BEV.
    No detectron2, no dataset directory.
    """

    def __init__(self, config):
        super(LiteExtractor, self).__init__(config)
        self.bev = bev_mod.BEV(config.image_settings["BEV_PATH"], mode="deploy")


# sg_autoencoder.__init__ does `RealEx.RealExtractor(self.config)`; the lookup
# happens at call time, so patching the attribute before the import suffices.
image_extractor.RealExtractor = LiteExtractor

from sgautoencoder.sg_autoencoder import sg_autoencoder             # noqa: E402


# --------------------------------------------------------------------------
# 4. Config
# --------------------------------------------------------------------------

def load_config(yaml_path=None):
    """
    Load `Config/pipeline_extraction.yaml`.  Its relative paths only resolve
    from `pipeline/`, so the BEV calibration path is rewritten to an absolute
    one if the relative form does not exist.
    """
    yaml_path = Path(yaml_path or GBSED_ROOT / "Config" / "pipeline_extraction.yaml")
    if not yaml_path.is_file():
        raise SystemExit("No such config file: %s" % yaml_path)
    cfg = configuration(str(yaml_path), from_function=True)

    bev_path = Path(cfg.image_settings["BEV_PATH"])
    if not bev_path.is_file():
        pkg = Path(image_extractor.__file__).resolve().parent   # .../scene_graph/extraction
        bev_path = pkg / "bev" / "bev.json"
    if not bev_path.is_file():
        raise SystemExit("Could not find bev.json (tried %s)" % bev_path)
    cfg.image_settings["BEV_PATH"] = str(bev_path)
    return cfg


def make_autoencoder(cfg):
    """The sg_autoencoder plus the BEV it carries, which Layer 1 also needs."""
    ae = sg_autoencoder(cfg)
    return ae, ae.sg_extraction_object.bev


def codebook_fingerprint(cfg):
    """
    ACTOR_NAMES and RELATION_NAMES are index spaces: nothing about them is
    transmitted, and node names are re-derived at decode from label indices.
    Encoder and decoder must agree exactly, so both ends record this hash.
    """
    s = repr(cfg.relation_extraction_settings["ACTOR_NAMES"]) + \
        repr(cfg.relation_extraction_settings["RELATION_NAMES"])
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# 5. Serialization -- byte-for-byte what pipeline.GBSED does
# --------------------------------------------------------------------------

def format_storage(labels, feature_nodes, L, comp_T):
    """Flatten the four arrays into one 1-D float16 vector, each section
    preceded by its own length.  Verbatim `GBSED._format_storage_`."""
    to_serialize = []
    to_serialize.append(len(labels))
    to_serialize.extend(labels)
    to_serialize.append(feature_nodes.size)
    to_serialize.extend(feature_nodes.ravel())
    to_serialize.append(L.size)
    to_serialize.extend(L)
    to_serialize.append(comp_T.size)
    to_serialize.extend(comp_T.ravel())
    return np.asarray(to_serialize, dtype=np.float16)


def format_loading(to_read):
    """Inverse of format_storage.  Verbatim `GBSED._format_loading_`.

    Note the final section takes `to_read[cur_idx:]` -- it ignores the declared
    comp_T length and consumes the remainder.  That is deliberate upstream (the
    Sionna path pads to a fixed block size); over a file transport the byte
    count is exact, so the two agree."""
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
    """float16 vector -> flat uint8 bit array.  Verbatim `_to_bits_array_`."""
    b = np_array.tobytes()
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def to_float_array(bits):
    """Flat bit array -> float16 vector.  Verbatim `_to_float_array_`."""
    b = np.packbits(bits)
    return np.frombuffer(b.tobytes(), np.float16)


def pack_to_bytes(packed_f16):
    """float16 payload -> the bytes that go in the .bin, via the bit array so
    the chain matches the wireless path exactly."""
    return np.packbits(to_bits_array(packed_f16)).tobytes()


def unpack_from_bytes(raw):
    """.bin bytes -> float16 payload."""
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return to_float_array(bits)


# --------------------------------------------------------------------------
# 6. Layer 1 -- image -> SceneGraph
# --------------------------------------------------------------------------

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

_DETECTOR = None      # loaded once and reused across a whole folder


def _get_detector(device):
    global _DETECTOR
    if _DETECTOR is None:
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        model.eval().to(device)
        _DETECTOR = (model, device)
    elif _DETECTOR[1] != device:
        _DETECTOR[0].to(device)
        _DETECTOR = (_DETECTOR[0], device)
    return _DETECTOR[0]


def detect_boxes(image_path, device="cpu", score_thresh=0.5):
    """
    Stand-in for detectron2's Mask R-CNN, which is not installed and whose
    DefaultPredictor defaults to CUDA.  Same (boxes, labels, image_size) triple.

    Downloads ~160 MB of COCO weights on first call.
    """
    import cv2
    import torch

    im = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit("Could not read image: %s" % image_path)
    im = cv2.resize(im, (1280, 720))                 # match the BEV calibration
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    model = _get_detector(device)
    with torch.no_grad():
        out = model([tensor.to(device)])[0]

    keep = out["scores"] >= score_thresh
    boxes = out["boxes"][keep].cpu()
    labels = [int(l) for l in out["labels"][keep].cpu()]
    scores = [float(s) for s in out["scores"][keep].cpu()]
    return boxes, labels, (720, 1280), scores


def demo_boxes():
    """Three cars in a 1280x720 frame -- the resolution bev.json is calibrated
    for.  Boxes are [left, top, right, bottom]; label 3 == 'car'."""
    import torch
    boxes = torch.tensor([
        [560.0, 380.0, 700.0, 480.0],    # ahead, same lane
        [300.0, 400.0, 460.0, 520.0],    # ahead-left
        [900.0, 390.0, 1050.0, 500.0],   # ahead-right
    ])
    return boxes, [3, 3, 3], (720, 1280), [1.0, 1.0, 1.0]


def scene_graph_from_boxes(cfg, bev, boxes, labels, image_size,
                           class_names=COCO_CLASS_NAMES):
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


# --------------------------------------------------------------------------
# 7. Layers 2-4 -- the round trip, as two callable halves
# --------------------------------------------------------------------------

def encode_scene_graph(ae, sg):
    """SceneGraph -> (payload bytes, detail dict).  Layers 2 and 3."""
    labels, feat_nodes_mat, T = ae.encode(sg)
    comp_T, L = ae.sem_compression(T)
    if comp_T.size == 0:
        raise ValueError(
            "scene graph has no active relations; nothing to compress or send")
    packed = format_storage(labels, feat_nodes_mat, L, comp_T)
    raw = pack_to_bytes(packed)
    return raw, {
        "labels": labels,
        "feature_nodes_matrix": feat_nodes_mat,
        "compressed_Tensor": comp_T,
        "indexes": [int(i) for i in L],
        "T_shape": list(T.shape),
        "n_bytes": len(raw),
        "n_float16": int(packed.size),
    }


def decode_payload(ae, raw):
    """Payload bytes -> (SceneGraph', detail dict).  Layer 4."""
    to_read = unpack_from_bytes(raw)
    labels, feature_nodes, L, comp_T = format_loading(to_read)
    sg = ae.decode(labels, feature_nodes, L, comp_T)
    return sg, {
        "labels": labels,
        "feature_nodes_matrix": feature_nodes,
        "compressed_Tensor": comp_T,
        "indexes": L,
    }


# --------------------------------------------------------------------------
# 8. Inspection and comparison
# --------------------------------------------------------------------------

def edge_set(sg):
    return sorted(
        (src.name, data["label"], dst.name) for src, dst, data in sg.g.edges(data=True)
    )


def node_list(sg):
    return [[n.name, n.label, int(n.value)] for n in sg.g.nodes]


def graph_summary(sg):
    """A JSON-serialisable description sufficient to score a reconstruction
    without re-running the detector."""
    return {
        "nodes": node_list(sg),
        "edges": [list(e) for e in edge_set(sg)],
        "n_nodes": sg.g.number_of_nodes(),
        "n_edges": sg.g.number_of_edges(),
    }


def describe_graph(sg, title):
    nodes = list(sg.g.nodes)
    print("\n%s" % title)
    print("  nodes (%d): %s" % (len(nodes), [n.name for n in nodes]))
    print("  edges (%d)" % sg.g.number_of_edges())


def compare_graphs(sg, rec_sg):
    """(ok, report_lines) comparing node names/labels and the relation multiset."""
    return compare_summary(graph_summary(sg), rec_sg)


def compare_summary(summary, rec_sg):
    """Same comparison, but against a stored `graph_summary` dict rather than a
    live SceneGraph -- so the decoder never needs the original image."""
    lines, ok = [], True

    orig_nodes = [tuple(n) for n in summary["nodes"]]
    rec_nodes = [tuple(n) for n in node_list(rec_sg)]
    if orig_nodes == rec_nodes:
        lines.append("  nodes            : identical (%d)" % len(orig_nodes))
    else:
        ok = False
        lines.append("  nodes            : DIFFER")
        lines.append("    original     : %s" % (orig_nodes,))
        lines.append("    reconstructed: %s" % (rec_nodes,))

    e0 = [tuple(e) for e in summary["edges"]]
    e1 = edge_set(rec_sg)
    if e0 == e1:
        lines.append("  relations        : identical (%d)" % len(e0))
    else:
        ok = False
        only_orig = [e for e in e0 if e not in e1]
        only_rec = [e for e in e1 if e not in e0]
        lines.append("  relations        : DIFFER (%d vs %d)" % (len(e0), len(e1)))
        lines.append("    missing in rec : %s" % (only_orig[:10],))
        lines.append("    extra in rec   : %s" % (only_rec[:10],))

    stats = {
        "n_nodes_orig": len(orig_nodes), "n_nodes_rec": len(rec_nodes),
        "n_edges_orig": len(e0), "n_edges_rec": len(e1),
        "n_edges_common": len(set(e0) & set(e1)),
        "nodes_match": orig_nodes == rec_nodes,
    }
    return ok, lines, stats


def maybe_visualize(sg, path):
    """Render with graphviz if pydot and the `dot` binary are available."""
    try:
        sg.visualize(str(path))
        return True
    except Exception as e:
        print("  could not render %s (%s: %s)" % (path, type(e).__name__, e))
        return False


def sha256_of(data):
    if isinstance(data, (str, os.PathLike)):
        data = Path(data).read_bytes()
    return hashlib.sha256(data).hexdigest()
