#!/usr/bin/env python3
"""
risk_assess.py -- run the downstream MRGCN risk classifier on a sequence of
received GBSED frames, reusing:
  - the same decode path gbsed_decode.py uses (gs.decode_payload)
  - the same SceneGraph -> tensor conversion roadscene2vec's own training
    pipeline uses (SceneGraph.get_real_image_node_embeddings /
    get_real_image_edge_embeddings, per roadscene2vec/scene_graph/scene_graph.py
    and roadscene2vec/data/dataset.py's process_real_image_graph_sequences)

Pipeline:
    received/*.bin --gs.decode_payload()--> SceneGraph (per frame)
                    --SceneGraph.get_real_image_*_embeddings()--> per-frame
                      (node_features, edge_index, edge_attr)
                    --torch_geometric Batch.from_data_list()--> one sequence
                    --MRGCN(config).forward(x, edge_index, edge_attr, batch)-->
                      risk prediction for the whole sequence

Why a sequence, not per-frame predictions: the config sets
temporal_type='lstm_attn' (task_type: sequence_classification). MRGCN's
forward() pools each frame (grouped by `batch` id) into one vector via
global_add_pool, then feeds the resulting sequence of per-frame vectors
through an LSTM+attention block. Running frames independently with
batch=zeros would silently give the LSTM a sequence of length 1 every call --
technically runs, but is not what the model was trained to do. This script
batches ALL frames from one run into a single sequence, matching training.

SECURITY: the checkpoint is loaded with weights_only=True, which refuses to
unpickle anything beyond tensors/basic containers. This specific checkpoint
was already statically inspected and confirmed to reference only
torch/collections internals (no os/subprocess/eval/exec). Re-run that check
on ANY new checkpoint before using this script with it:

    python -c "
    import zipfile, pickletools
    with zipfile.ZipFile('<path>.pt') as z:
        names = [n for n in z.namelist() if n.endswith('data.pkl')]
        with z.open(names[0]) as f:
            pickletools.dis(f.read())
    " | grep -iE 'GLOBAL|REDUCE'

Usage:
    python risk_assess.py \
        --received /home/opp_env/default_workspace/veins/src/veins/modules/application/gbsed/GBSEDApp/received \
        --meta /home/opp_env/default_workspace/gbsed/scene_data \
        --checkpoint /home/opp_env/default_workspace/gbsed/checkpoints/1043_task_oriented_model.pt \
        --config /home/opp_env/default_workspace/gbsed/Config/pipeline_learning.yaml
"""
import argparse
import json
import sys
import types
from pathlib import Path

import yaml
import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # gbsed/ repo root
sys.path.insert(0, "/home/opp_env/default_workspace/roadscene2vec")

import gbsed_semantic as gs  # noqa: E402  -- same import gbsed_decode.py uses
from roadscene2vec.learning.model.mrgcn import MRGCN  # noqa: E402


# --------------------------------------------------------------------------
# 1. Safe checkpoint loading
# --------------------------------------------------------------------------
def load_checkpoint_safely(path):
    sd = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError(f"Expected a state_dict (dict), got {type(sd)}")
    return sd


# --------------------------------------------------------------------------
# 2. Build the model from pipeline_learning.yaml, forced onto CPU
# --------------------------------------------------------------------------
def build_model(config_path, device="cpu"):
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    model_configuration = dict(raw["model_configuration"])
    model_configuration["device"] = device  # yaml ships with 'cuda'; force CPU

    config = types.SimpleNamespace(model_configuration=model_configuration)
    model = MRGCN(config)
    return model, raw


# --------------------------------------------------------------------------
# 3. Find each frame's received .bin, same lookup gbsed_decode.py uses
# --------------------------------------------------------------------------
def find_received(received_dir, meta):
    for c in (received_dir / ("received_" + meta["bin"]), received_dir / meta["bin"]):
        if c.is_file():
            return c
    return None


# --------------------------------------------------------------------------
# 4. SceneGraph -> PyG Data, using the SAME methods the training pipeline
#    uses (roadscene2vec/data/dataset.py: process_real_image_graph_sequences)
# --------------------------------------------------------------------------
def scenegraph_to_pyg_data(sg, num_actor_classes):
    # feature_list defines the one-hot column order. Training builds this
    # as "type_<i>" for each possible actor class index (see
    # scene_graph.py: row['type_'+str(node.value)] = 1). This must be built
    # the SAME way here or the one-hot columns will misalign -- the exact
    # failure mode the project's invariant 2 warns about.
    feature_list = ["type_%d" % i for i in range(num_actor_classes)]

    node_name2idx = {node: idx for idx, node in enumerate(sg.g.nodes)}

    x = sg.get_real_image_node_embeddings(feature_list)          # FloatTensor [N, num_actor_classes]
    edge_index, edge_attr = sg.get_real_image_edge_embeddings(node_name2idx)  # LongTensor [2,E], LongTensor [E]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--received", required=True,
                     help="path to GBSEDApp/received directory")
    ap.add_argument("--meta", required=True,
                     help="path to scene_data directory (holds *.meta.json)")
    ap.add_argument("--checkpoint", required=True, help="path to .pt checkpoint")
    ap.add_argument("--config", required=True, help="path to pipeline_learning.yaml")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    print("Loading checkpoint (restricted mode)...")
    state_dict = load_checkpoint_safely(args.checkpoint)

    print("Building model from config...")
    model, raw_cfg = build_model(args.config, device=args.device)
    num_actor_classes = raw_cfg["model_configuration"]["num_of_classes"]

    print("Loading weights into model...")
    model.load_state_dict(state_dict)
    model.eval()

    meta_dir = Path(args.meta)
    received_dir = Path(args.received)
    meta_files = sorted(meta_dir.glob("*.meta.json"), key=lambda p: json.loads(p.read_text())["frame"])
    if not meta_files:
        raise SystemExit(f"No *.meta.json files found in {meta_dir}")

    cfg = gs.load_config()  # uses the default: Config/pipeline_extraction.yaml
    ae, _bev = gs.make_autoencoder(cfg)

    print(f"\nDecoding {len(meta_files)} frame(s) and building the sequence...")
    frame_data = []
    skipped = []
    for mf in meta_files:
        meta = json.loads(mf.read_text())
        bin_path = find_received(received_dir, meta)
        if bin_path is None:
            print(f"  {meta['stem']}: LOST (never received) -- excluded from sequence")
            skipped.append(meta["stem"])
            continue

        raw = bin_path.read_bytes()
        try:
            rec_sg, _detail = gs.decode_payload(ae, raw)
            data = scenegraph_to_pyg_data(rec_sg, num_actor_classes)
        except Exception as e:
            print(f"  {meta['stem']}: FAILED to decode/convert ({type(e).__name__}: {e}) -- excluded")
            skipped.append(meta["stem"])
            continue

        print(f"  {meta['stem']}: ok  (nodes={data.x.size(0)}, edges={data.edge_index.size(1)})")
        frame_data.append(data)

    if not frame_data:
        raise SystemExit("\nNo frames could be decoded -- nothing to run inference on.")

    # Batch.from_data_list assigns each Data object's nodes a shared `batch`
    # id (0, 1, 2, ... one per frame) -- exactly the grouping
    # global_add_pool needs to turn per-node features back into one
    # per-frame vector before the LSTM sees it.
    sequence = Batch.from_data_list(frame_data)

    print(f"\nRunning inference on the {len(frame_data)}-frame sequence "
          f"({len(skipped)} frame(s) excluded: {skipped or 'none'})...")
    with torch.no_grad():
        log_probs, attn_weights = model(sequence.x, sequence.edge_index,
                                         sequence.edge_attr, sequence.batch)
        pred = log_probs.argmax(dim=-1).tolist()
        probs = log_probs.exp().tolist()

    result = {
        "n_frames_used": len(frame_data),
        "n_frames_skipped": len(skipped),
        "skipped_frames": skipped,
        "prediction": pred,
        "class_probabilities": probs,
    }

    out_path = meta_dir.parent / "risk_prediction.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nPrediction: {pred}")
    print(f"Class probabilities: {probs}")
    print(f"Wrote result to {out_path}")


if __name__ == "__main__":
    main()