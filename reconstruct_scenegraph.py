#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Official GBSED Scene Graph Reconstructor
Reconstructs the real SceneGraph data directly from .bin payload files.
"""

import os
import sys
import json
import struct
from pathlib import Path
import numpy as np

# Official GBSED configuration definitions
ACTOR_NAMES = ["ego_car", "car", "moto", "bicycle", "ped", "lane", "light", "sign", "road"]
RELATION_NAMES = [
    "isIn", "inDFrontOf", "inSFrontOf", "atDRearOf", "atSRearOf", 
    "toLeftOf", "toRightOf", "near_coll", "super_near", "very_near", "near", "visible"
]
FEATURE_KEYS = [
    'left', 'top', 'right', 'bottom', 
    'location_x', 'location_y', 'rel_location_x', 'rel_location_y', 'distance_abs'
]

def format_loading(to_read):
    """
    Official GBSED deserialization from 1D float16 array (pipeline.py lines 130-186).
    """
    cur_idx = 0
    
    # 1. Labels
    nb = int(to_read[cur_idx]); cur_idx += 1
    end_idx = cur_idx + nb
    labels = to_read[cur_idx : end_idx]
    
    # 2. Features
    cur_idx = end_idx
    nb = int(to_read[cur_idx]); cur_idx += 1
    end_idx = cur_idx + nb
    features = to_read[cur_idx : end_idx]
    
    # 3. Active relation indices L
    cur_idx = end_idx
    nb = int(to_read[cur_idx]); cur_idx += 1
    end_idx = cur_idx + nb
    L = to_read[cur_idx : end_idx]
    
    # 4. Compressed tensor comp_T
    cur_idx = end_idx
    nb = int(to_read[cur_idx]); cur_idx += 1
    end_idx = cur_idx + nb
    comp_T = to_read[cur_idx : end_idx]
    
    labels = [int(i) for i in labels]
    feature_nodes = features.reshape(((len(labels)), -1))
    L = [int(i) for i in L]
    comp_T = comp_T.reshape((len(L), len(labels), len(labels)))
    
    return labels, feature_nodes, L, comp_T

def sem_decompression(comp_T, L, num_relations=len(RELATION_NAMES)):
    """
    Official GBSED inverse semantic compression (sg_autoencoder.py lines 111-134).
    """
    shape = comp_T.shape[1:]
    T = np.zeros((num_relations, shape[0], shape[1]), dtype=comp_T.dtype)
    for i in range(num_relations):
        if i in L:
            pos = L.index(i)
            T[i] = comp_T[pos].copy()
    return T

def decode_scenegraph(bin_file_path):
    """
    Decodes the .bin file and reconstructs the complete SceneGraph data structures.
    """
    bin_file_path = Path(bin_file_path)
    with open(bin_file_path, "rb") as f:
        raw_bytes = f.read()
        
    float_array = np.frombuffer(raw_bytes, dtype=np.float16)
    labels, feature_nodes, L, comp_T = format_loading(float_array)
    T = sem_decompression(comp_T, L)
    
    # Reconstruct Node Names (sg_autoencoder._get_all_node_names)
    node_names = []
    for i, l_idx in enumerate(labels):
        name = ACTOR_NAMES[l_idx]
        if name == "ego_car":
            to_append = "ego car"
        elif name == "road":
            to_append = "Root Road"
        elif name == "lane":
            if "Right Lane" in node_names:
                to_append = "Middle Lane"
            else:
                to_append = "Right Lane" if "Left Lane" in node_names else "Left Lane"
        else:
            suffix = int(feature_nodes[i, -1])
            to_append = f"{name}_{suffix}"
        node_names.append(to_append)
        
    # Reconstruct Node Attributes (sg_autoencoder._get_node)
    nodes_list = []
    for i, name in enumerate(node_names):
        l_idx = labels[i]
        actor_type = ACTOR_NAMES[l_idx]
        attr = {}
        if "_" in name:
            for k_idx, k in enumerate(FEATURE_KEYS):
                attr[k] = float(feature_nodes[i, k_idx])
        elif "road" in name.lower():
            attr["name"] = "Root Road"
        elif "ego" in name.lower():
            attr["location_x"] = float(feature_nodes[i, 0])
            attr["location_y"] = float(feature_nodes[i, 1])
        nodes_list.append({
            "id": i,
            "name": name,
            "label": actor_type,
            "label_idx": l_idx,
            "attributes": attr
        })
        
    # Reconstruct Directed Relations
    edges_list = []
    num_nodes = len(node_names)
    for r in range(len(RELATION_NAMES)):
        for j in range(num_nodes):
            for k in range(num_nodes):
                if T[r, j, k] != 0:
                    edges_list.append({
                        "source": node_names[j],
                        "source_id": j,
                        "relation": RELATION_NAMES[r],
                        "relation_idx": r,
                        "target": node_names[k],
                        "target_id": k
                    })
                    
    return {
        "source_file": str(bin_file_path),
        "payload_bytes": len(raw_bytes),
        "num_nodes": len(nodes_list),
        "num_relations": len(edges_list),
        "active_relation_types": [RELATION_NAMES[i] for i in L],
        "nodes": nodes_list,
        "edges": edges_list,
        "raw_tensors": {
            "labels": labels,
            "L": L,
            "feature_matrix_shape": list(feature_nodes.shape),
            "compressed_tensor_shape": list(comp_T.shape),
            "full_tensor_shape": list(T.shape)
        }
    }

def save_reconstructed_outputs(decoded_data, output_json_path, output_txt_path):
    output_json_path = Path(output_json_path)
    output_txt_path = Path(output_txt_path)
    
    # 1. Save JSON dump
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(decoded_data, f, indent=2)
    print(f"Saved Reconstructed SceneGraph JSON: {output_json_path}")
    
    # 2. Save readable text report
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"GBSED RECONSTRUCTED SCENE GRAPH REPORT\n")
        f.write(f"Source Binary Payload: {decoded_data['source_file']} ({decoded_data['payload_bytes']} bytes)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"1. RECONSTRUCTED NODES ({decoded_data['num_nodes']} total):\n")
        f.write("-" * 80 + "\n")
        for node in decoded_data["nodes"]:
            f.write(f"  [Node {node['id']}] Name: '{node['name']}' | Type: '{node['label']}' (Index: {node['label_idx']})\n")
            if node["attributes"]:
                f.write(f"             Attributes: {node['attributes']}\n")
        f.write("\n")
        
        f.write(f"2. RECONSTRUCTED DIRECTED RELATIONS ({decoded_data['num_relations']} total edges):\n")
        f.write("-" * 80 + "\n")
        for edge in decoded_data["edges"]:
            f.write(f"  ({edge['source']}) ---[{edge['relation']}]---> ({edge['target']})\n")
        f.write("\n")
        
        f.write("3. ACTIVE RELATION TYPES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  {decoded_data['active_relation_types']}\n\n")
        f.write("=" * 80 + "\n")
    print(f"Saved Reconstructed SceneGraph Text: {output_txt_path}")

if __name__ == "__main__":
    base_dir = Path("c:/Users/ACER/Desktop/thesis/gbsed project/gbsed")
    
    # 1. Reconstruct received file
    received_bin = base_dir / "received" / "received_scene_0000.bin"
    rec_data = decode_scenegraph(received_bin)
    save_reconstructed_outputs(
        rec_data,
        base_dir / "received" / "reconstructed_scene_graph.json",
        base_dir / "received" / "reconstructed_scene_graph.txt"
    )
    
    # 2. Reconstruct original sent file
    sent_bin = base_dir / "scene_data" / "scene_0000.bin"
    sent_data = decode_scenegraph(sent_bin)
    save_reconstructed_outputs(
        sent_data,
        base_dir / "scene_data" / "reconstructed_scene_graph.json",
        base_dir / "scene_data" / "reconstructed_scene_graph.txt"
    )
