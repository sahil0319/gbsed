# Reading guide — understanding `load_model.ipynb`

The notebook is roughly 15% glue code written to make things run on this machine and
85% calls into two existing codebases. You only need to understand the second part.

---

## What the notebook actually contains

| cells | what it is | do you need to read the source? |
|---|---|---|
| 2, 4, 6 | path setup, detectron2 stub, `LiteExtractor` | No — plumbing to make imports work without a GPU |
| 8 | config loading | Barely — reads a yaml, fixes one path |
| 10 | serialization helpers | **Yes — copied verbatim from `pipeline.py`** |
| 12 | box → SceneGraph, torchvision detector | Skim the glue; read what it *calls* |
| 14, 16 | comparison helpers, the three tests | Glue, self-contained |
| 18–28 | execution | — |

So the real implementation lives in two files of this project and about five of
roadscene2vec's.

---

## Tier 1 — the semantic core (read these properly)

### `sgautoencoder/sg_autoencoder.py` — 203 lines, the heart of it

Read in this order:

1. **`encode()` — lines 136-182.** Walks `sg.g.adjacency()`, builds the
   `[relations × nodes × nodes]` tensor `T`, the label list, and the float16 feature
   matrix. Everything downstream is determined here.
2. **`sem_compression()` — lines 84-109** and **`sem_decompression()` — lines 111-134.**
   Drop and restore all-zero relation slices. Twenty lines total, and this is the
   paper's "semantic compression".
3. **`decode()` — lines 185-203.** Rebuilds a `SceneGraph` from the four arrays.
4. **`_get_all_node_names()` — lines 36-55** and **`_get_node()` — lines 57-82.**
   The trickiest part: node *names* are never transmitted, they are **re-derived** from
   label indices plus a positional heuristic. Understand why the lane branch checks
   `if "Right Lane" in names`.

### `pipeline/pipeline.py` — read only four short methods

The ones notebook cell 10 copies:

- `_format_storage_` — lines 97-128
- `_format_loading_` — lines 130-186
- `_to_bits_array_` — lines 219-222
- `_to_float_array_` — lines 224-227

Then read `_process_sg_` (lines 229-255) and `_sg_reconstruction_` (lines 257-289) to
see how the author chains them — those two are exactly what tests B and C reproduce
without the channel.

---

## Tier 2 — how a SceneGraph gets built (roadscene2vec)

All paths below are relative to `../roadscene2vec/roadscene2vec/`.

### `scene_graph/scene_graph.py`

Read the `platform == "image"` branch of `__init__` (lines 27-58), then
`get_nodes_from_bboxes()` (lines 61-109), then the two-line `add_node` (112-121) and
`add_relation` (131-138). This file decides node insertion order and stamps `value=` on
every edge — the two things `encode()` reads.

### `scene_graph/nodes.py`

10 lines. `Node(name, attr, label, value)`. Read it first, actually; it takes 30 seconds
and the rest makes more sense afterwards.

### `scene_graph/relation_extractor.py`

Where the 21 edges in the test graph come from:

- `extract_relative_lanes` — lines 50-66
- `add_mapping_to_relative_lanes` — lines 69-90
- `extract_semantic_relations` — lines 92-95
- `create_proximity_relations` — lines 157-161
- `extract_directional_relation` — lines 163-212

### `Config/pipeline_extraction.yaml` (in this repo)

Read it alongside the above. `ACTOR_NAMES` and `RELATION_NAMES` are *index spaces*: a
label of `8` means nothing except "position 8 in ACTOR_NAMES". Reorder those lists and
every previously-encoded graph decodes wrong.

### `scene_graph/extraction/bev/bev.py`

Just three methods — how pixels become feet:

- `apply_depth_estimation` — lines 123-124
- `get_projected_point` — lines 126-129
- `compute_homography_matrix` — lines 131-151

### `scene_graph/extraction/image_extractor.py`

Read `__init__` (lines 21-40) and `get_bounding_boxes` (lines 115-127) only, to see
precisely what `LiteExtractor` drops and what `detect_boxes()` replaces. Plus
`scene_graph/extraction/extractor.py` — 15 lines, the base class.

---

## Skip entirely for now

`Communication/e2emodel.py`, `Communication/receiver.py`, and `pipeline.py`'s
`_prepare_bits_for_model_`, `_recover_original_bits_`, `sg_transmission`, `e2e`, `main`,
plus everything under `roadscene2vec/learning/`. That is Layer 3 (the wireless channel)
and the downstream classifier.

---

## Three ideas that make the rest click

**1. There are two unrelated `value` fields.**
`node.value` = index into `ACTOR_NAMES` (becomes `labels`).
`edge['value']` = index into `RELATION_NAMES` (becomes the first axis of `T`).
`encode()` reads both; confusing them will make the tensor layout look arbitrary.

**2. Node identity is positional, not nominal.**
`T[r, j, k]` refers to nodes by their index in `list(sg.g.adjacency())`, and names are
regenerated at decode time. The round trip works because insertion order is
deterministic: Root Road → ego → Left/Right/Middle Lane → cars.

**3. The feature matrix is positional too.**
`encode()` does `list(node.attr.values())` — column meaning comes from Python dict
insertion order in `get_nodes_from_bboxes` — and `_get_node()` (lines 61-71) hardcodes
`features[0]`…`features[8]` to read them back. Add an attribute anywhere in
`get_nodes_from_bboxes` and decode silently misreads every column after it.

---

## A good exercise

In notebook cell 20, print:

```python
[(n.name, n.value, n.attr) for n in sg.g.nodes]
list(sg.g.edges(data=True))
```

Then trace `car_0` by hand through `encode()` into `T` and `feat_nodes_mat`, and back
out through `_get_all_node_names` and `_get_node`. An hour on that one node teaches more
than reading all seven files linearly.
