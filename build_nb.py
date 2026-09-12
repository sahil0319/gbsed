import nbformat as nbf
import os
import json

nb = nbf.v4.new_notebook()
cells = []

def add_md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def add_code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

# 1. Experiment Configuration
add_md("""
# GBSED End-to-End Semantic Communication over Wireless Channel

This notebook integrates the GBSED semantic communication pipeline with
the repository's Sionna-based physical-layer communication model.

Pipeline:

SceneGraph
→ Semantic Encoding
→ Semantic Compression
→ Storage Formatting
→ Bit Conversion
→ MIMOE2EModel
→ Bit Recovery
→ Semantic Reconstruction
→ SceneGraph Reconstruction

The first experiment is intentionally limited to one SceneGraph at
Eb/N0 = 20 dB because the current machine is CPU-only.
""")

add_code("""
RANDOM_SEED = 42
EBNO_DB = 20.0
BATCH_SIZE = 1

import numpy as np

np.random.seed(RANDOM_SEED)
""")

# 2. Environment and CPU Verification
add_md("## 2. Environment and CPU Verification")
add_code("""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import sionna

print("TensorFlow:", tf.__version__)
print("Sionna:", sionna.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print("CPU devices:", tf.config.list_physical_devices("CPU"))

if not tf.config.list_physical_devices("GPU"):
    print("Running GBSED PHY on CPU.")
else:
    print("GPU detected.")
""")

# 3. Import GBSED Components
add_md("## 3. Import GBSED Components")
add_code("""
import sys
import types
from pathlib import Path
import pickle as pkl
import platform

try:
    GBSED_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    GBSED_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

if not (GBSED_ROOT / "sgautoencoder").is_dir():
    GBSED_ROOT = Path("E:/Project/Project/gbsed")
    
def _find_roadscene2vec():
    candidates = [
        GBSED_ROOT.parent / "roadscene2vec",
        GBSED_ROOT / "roadscene2vec",
    ]
    for root in candidates:
        if (root / "roadscene2vec" / "__init__.py").is_file():
            return root
    try:
        import roadscene2vec 
        return None
    except ImportError:
        raise SystemExit("Could not locate roadscene2vec.")

_R2V_ROOT = _find_roadscene2vec()
if _R2V_ROOT is not None and str(_R2V_ROOT) not in sys.path:
    sys.path.insert(0, str(_R2V_ROOT))
if str(GBSED_ROOT) not in sys.path:
    sys.path.insert(0, str(GBSED_ROOT))

def _stub_detectron2():
    try:
        import detectron2
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
    def _unavailable(*_a, **_kw): pass
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

from roadscene2vec.util.config_parser import configuration
from roadscene2vec.scene_graph.scene_graph import SceneGraph
from roadscene2vec.scene_graph.extraction import extractor as base_ex
from roadscene2vec.scene_graph.extraction import image_extractor
from roadscene2vec.scene_graph.extraction.bev import bev as bev_mod

class LiteExtractor(base_ex.Extractor):
    def __init__(self, config):
        super(LiteExtractor, self).__init__(config)
        self.bev = bev_mod.BEV(config.image_settings["BEV_PATH"], mode="deploy")

image_extractor.RealExtractor = LiteExtractor

from Communication.e2emodel import MIMOE2EModel
from sgautoencoder.sg_autoencoder import sg_autoencoder

# Helpers
def load_config(yaml_path=None):
    yaml_path = Path(yaml_path or GBSED_ROOT / "Config" / "pipeline_extraction.yaml")
    cfg = configuration(str(yaml_path), from_function=True)
    bev_path = Path(cfg.image_settings["BEV_PATH"])
    if not bev_path.is_file():
        pkg = Path(image_extractor.__file__).resolve().parent
        bev_path = pkg / "bev" / "bev.json"
    cfg.image_settings["BEV_PATH"] = str(bev_path)
    return cfg

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
    return np.asarray(to_serialize, dtype=np.float16)

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

def prepare_bits_for_model(input_bits_1d, expected_bits):
    if len(input_bits_1d) > expected_bits:
        prepared_bits = input_bits_1d[:expected_bits]
    elif len(input_bits_1d) < expected_bits:
        padding_size = expected_bits - len(input_bits_1d)
        prepared_bits = np.pad(input_bits_1d, (0, padding_size), 'constant', constant_values=0)
    else:
        prepared_bits = input_bits_1d
    return tf.constant(prepared_bits, dtype=tf.int32)

def recover_original_bits(decoded_bits, original_length):
    decoded_bits_flat = tf.reshape(decoded_bits, [-1])
    return decoded_bits_flat[:original_length]
    
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
    import torch
    boxes = torch.tensor([
        [560.0, 380.0, 700.0, 480.0],
        [300.0, 400.0, 460.0, 520.0],
        [900.0, 390.0, 1050.0, 500.0],
    ])
    return boxes, [3, 3, 3], (720, 1280)

def describe_graph(sg, title):
    nodes = list(sg.g.nodes)
    print("\\n%s" % title)
    print("  nodes (%d): %s" % (len(nodes), [n.name for n in nodes]))
    print("  edges (%d)" % sg.g.number_of_edges())

def edge_set(sg):
    return sorted((src.name, data["label"], dst.name) for src, dst, data in sg.g.edges(data=True))

def compare_graphs(sg, rec_sg):
    lines, ok = [], True
    orig_nodes = [(n.name, n.label, n.value) for n in sg.g.nodes]
    rec_nodes = [(n.name, n.label, n.value) for n in rec_sg.g.nodes]
    if orig_nodes == rec_nodes:
        lines.append("  nodes          : identical (%d)" % len(orig_nodes))
        node_accuracy = 1.0
    else:
        ok = False
        lines.append("  nodes          : DIFFER")
        # simple accuracy metric for demo
        matches = sum(1 for n in orig_nodes if n in rec_nodes)
        node_accuracy = matches / max(len(orig_nodes), 1)
        
    e0, e1 = edge_set(sg), edge_set(rec_sg)
    if e0 == e1:
        lines.append("  relations      : identical (%d)" % len(e0))
        rel_accuracy = 1.0
    else:
        ok = False
        lines.append("  relations      : DIFFER (%d vs %d)" % (len(e0), len(e1)))
        matches = sum(1 for e in e0 if e in e1)
        rel_accuracy = matches / max(len(e0), 1)
        
    graph_match = float(ok)
    return ok, lines, node_accuracy, rel_accuracy, graph_match
""")

# 4. Load GBSED Semantic Model
add_md("## 4. Load GBSED Semantic Model")
add_code("""
cfg = load_config()
sg_autoencoder = sg_autoencoder(cfg)
print("GBSED semantic model loaded successfully.")
""")

# 5. Load/Create Test SceneGraph
add_md("## 5. Load/Create Test SceneGraph")
add_code("""
bev = sg_autoencoder.sg_extraction_object.bev
boxes, labels_, image_size = demo_boxes()
scene_graph = scene_graph_from_boxes(cfg, bev, boxes, labels_, image_size)
describe_graph(scene_graph, "Original SceneGraph")
""")

# 6. Semantic Encoding
add_md("## 6. Semantic Encoding")
add_code("""
labels, feature_nodes, T = sg_autoencoder.encode(scene_graph)
semantic_encoded = feature_nodes # for shape printing just to have something, or T
print("Original SceneGraph")
print("        ↓")
print("Encoded semantic representation")
print("Encoded semantic shape:", T.shape)
""")

# 7. Semantic Compression
add_md("## 7. Semantic Compression")
add_code("""
comp_T, L = sg_autoencoder.sem_compression(T)
semantic_compressed = comp_T
print("Semantic compression completed.")
print("Compressed semantic shape:", semantic_compressed.shape)
""")


# 8. Storage Formatting
add_md("## 8. Storage Formatting")
add_code("""
storage_representation = format_storage(labels, feature_nodes, L, comp_T)
print("Storage representation shape:", storage_representation.shape)
""")

# 9. Bit Conversion
add_md("## 9. Convert Semantic Payload to Bits")
add_code("""
tx_bits = to_bits_array(storage_representation)
tx_bits = np.asarray(tx_bits).astype(np.int32).flatten()
print("GBSED payload length:", len(tx_bits))
print("First 32 bits:", tx_bits[:32])
""")

# 10. PHY Configuration
add_md("## 10. PHY Configuration")
add_code("""
phy_model = MIMOE2EModel()
# build model
phy_model(1, tf.constant(0.0, tf.float32))

print("PHY configuration")
print("-" * 50)
print("n:", phy_model.n)
print("k:", int(phy_model.k))
print("Users:", phy_model.num_ut)
print("TX antennas:", phy_model.num_tx_ant)
print("Bits/QAM symbol:", phy_model.num_bits_per_symbol)
print("Code rate:", phy_model.coderate)

required_phy_bits = (
    BATCH_SIZE
    * phy_model.num_ut
    * phy_model.num_tx_ant
    * int(phy_model.k)
)
print("Required PHY input bits:", required_phy_bits)
print("GBSED payload bits:", len(tx_bits))
""")

# 11. PHY Payload Compatibility Check
add_md("## 11. PHY Payload Compatibility Check")
add_code("""
# GBSED semantic payload -> repository-defined payload formatting -> PHY-compatible information bits
print("Mapping GBSED payload to PHY input using zero-padding as defined in pipeline.py")
""")

# 12. Neural Receiver Verification
add_md("## 12. Neural Receiver Verification")
add_code("""
# Load pretrained weights as done in pipeline.py
model_weights_path = GBSED_ROOT / "Communication" / "weights" / "neural_rx_ofdm_mimo_cdl_final.h5"
if model_weights_path.exists():
    with open(model_weights_path, 'rb') as f:
        weights = pkl.load(f)
    for i, w in enumerate(weights):
        phy_model.neural_rx.weights[i].assign(w)
    print("NeuralReceiver pretrained weights loaded successfully.")
else:
    print("WARNING: pretrained weights not found at", model_weights_path)
""")

# 13. Standalone PHY Smoke Test
add_md("## 13. Standalone PHY Smoke Test")
add_code("""
num_test_bits = required_phy_bits
test_bits = tf.random.uniform(
    [num_test_bits],
    minval=0,
    maxval=2,
    dtype=tf.int32
)

ebno_db = tf.constant(EBNO_DB, dtype=tf.float32)
# The e2emodel expects input_bits if provided
b, b_hat = phy_model(BATCH_SIZE, ebno_db, test_bits)

tx_phy_bits = tf.reshape(b, [-1]).numpy().astype(np.int32)
rx_phy_bits = tf.reshape(b_hat, [-1]).numpy().astype(np.int32)

smoke_bit_errors = np.sum(tx_phy_bits != rx_phy_bits)
smoke_ber = smoke_bit_errors / len(tx_phy_bits)

print("PHY smoke test")
print("=" * 50)
print("Eb/N0:", EBNO_DB, "dB")
print("Total bits:", len(tx_phy_bits))
print("Bit errors:", smoke_bit_errors)
print("BER:", smoke_ber)
""")

# 14. End-to-End GBSED Transmission
add_md("## 14. Actual GBSED Payload Through Physical Channel")
add_code("""
phy_input_bits = prepare_bits_for_model(tx_bits, required_phy_bits)

b, b_hat = phy_model(
    BATCH_SIZE,
    ebno_db,
    phy_input_bits
)
print("Actual GBSED transmission completed.")
""")

# 15. PHY Bit Recovery
add_md("## 15. PHY Bit Recovery")
add_code("""
rx_bits = tf.reshape(b_hat, [-1]).numpy().astype(np.int32)
actual_tx_bits = tf.reshape(b, [-1]).numpy().astype(np.int32)

bit_errors = np.sum(actual_tx_bits != rx_bits)
ber = bit_errors / len(actual_tx_bits)

print("Actual GBSED PHY transmission")
print("-" * 32)
print("Eb/N0:", EBNO_DB)
print("Input bits:", len(actual_tx_bits))
print("Recovered bits:", len(rx_bits))
print("Bit errors:", bit_errors)
print("BER:", ber)
""")

# 16. Reconstruct Semantic Representation
add_md("## 16. Reverse the Bit Conversion")
add_code("""
recovered_bits = recover_original_bits(b_hat, len(tx_bits)).numpy().astype(np.uint8)
recovered_storage = to_float_array(recovered_bits)
print("Transmitter: float representation -> bits")
print("Receiver: bits -> float representation")
""")

# 17. GBSED Semantic Decoding
add_md("## 17. Reverse Storage Formatting")
add_code("""
rec_labels, rec_feat, rec_L, rec_comp_T = format_loading(recovered_storage)
print("Recovered semantic shape:", rec_comp_T.shape)
""")

add_md("## 18. GBSED Semantic Decoder")
add_code("""
reconstructed_graph = sg_autoencoder.decode(rec_labels, rec_feat, list(rec_L), rec_comp_T)
describe_graph(reconstructed_graph, "Reconstructed SceneGraph")
""")

# 18. SceneGraph Comparison
add_md("## 19. SceneGraph Comparison")
add_code("""
ok, lines, node_acc, rel_acc, graph_match = compare_graphs(scene_graph, reconstructed_graph)
for line in lines:
    print(line)
""")

add_md("## 20. Multi-Level Metrics")
add_code("""
# Level 1 - Physical Layer
# Handled in bit_errors/ber

# Level 2 - Semantic Representation
mse_features = np.mean((feature_nodes - rec_feat)**2)
mse_t = np.mean((comp_T - rec_comp_T)**2)

# Level 3 - SceneGraph (node_acc, rel_acc)

print("Semantic Feature MSE:", mse_features)
print("Semantic Topology MSE (comp_T):", mse_t)
""")

add_md("## 21. Final Test-D Report")
add_code("""
results = {
    "Eb/N0 (dB)": EBNO_DB,
    "PHY input bits": len(actual_tx_bits),
    "PHY bit errors": int(bit_errors),
    "BER": float(ber),
    "Semantic Feature MSE": float(mse_features),
    "Semantic Topology MSE": float(mse_t),
    "Node accuracy": float(node_acc),
    "Relation accuracy": float(rel_acc),
    "Graph match": float(graph_match)
}

print("========================================================")
print("GBSED END-TO-END RESULT")
print("========================================================")
print(f"Eb/N0                 : {results['Eb/N0 (dB)']} dB\\n")

print("PHY")
print(f"  Information bits   : {results['PHY input bits']}")
print(f"  Bit errors         : {results['PHY bit errors']}")
print(f"  BER                : {results['BER']:.6f}\\n")

print("SEMANTIC")
print(f"  Feature MSE        : {results['Semantic Feature MSE']:.6f}")
print(f"  Topology MSE       : {results['Semantic Topology MSE']:.6f}\\n")

print("SCENEGRAPH")
print(f"  Node accuracy      : {results['Node accuracy']:.4f}")
print(f"  Relation accuracy  : {results['Relation accuracy']:.4f}")
print(f"  Graph match        : {results['Graph match']:.1f}\\n")

print("TEST D —", "PASS" if bool(graph_match) else "FAIL")
""")

nb['cells'] = cells
with open('e:/Project/Project/gbsed/notebooks/gbsed_end_to_end_channel.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
