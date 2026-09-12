"""
Legacy PyTorch Geometric compatibility shim.

roadscene2vec/roadscene2vec/learning/model/rgcn_sag_pooling.py does:

    from torch_geometric.nn.pool.topk_pool import topk, filter_adj

These two free functions existed in torch_geometric.nn.pool.topk_pool up
through PyG ~2.3 and were removed when pooling was refactored around
SelectTopK/FilterEdges (PyG 2.4+). This module is a line-for-line port of
the original PyG 1.6.1 implementation (see
https://pytorch-geometric.readthedocs.io/en/1.6.1/_modules/torch_geometric/nn/pool/topk_pool.html),
with only the torch_scatter calls (scatter_add/scatter_max) rewritten
using native torch ops, so no separate torch_scatter dependency is needed.

We deliberately do NOT wrap the new SelectTopK/FilterEdges classes: their
internal tie-breaking and indexing differ enough from the pre-2.4
algorithm that a checkpoint trained against the old `topk`/`filter_adj`
could silently pool different nodes/edges under the new classes. This
port reproduces the exact old arithmetic instead.

Nothing in the roadscene2vec source tree is imported or modified here.
Call install() once, before anything imports
roadscene2vec.learning.model.mrgcn (which imports rgcn_sag_pooling).
"""
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes


def _topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        scores_max = torch.zeros_like(x).scatter_reduce_(
            0, batch, x, reduce="amax", include_self=False
        )[batch] - tol
        scores_min = scores_max.clamp(max=min_score)
        perm = torch.nonzero(x > scores_min).view(-1)
    else:
        num_nodes = torch.zeros(int(batch.max()) + 1, dtype=torch.long, device=x.device)
        num_nodes.scatter_add_(0, batch, torch.ones_like(batch))
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max().item())
        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0
        )
        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes,), -2.0)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)
        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
        if isinstance(ratio, int) and (num_nodes < ratio).sum() > 0:
            k = num_nodes.clamp(max=ratio)

        mask = [
            torch.arange(int(k[i]), dtype=torch.long, device=x.device) + i * max_num_nodes
            for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)
        perm = perm[mask]
    return perm


def _filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    mask = perm.new_full((num_nodes,), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    keep = (row >= 0) & (col >= 0)
    row, col = row[keep], col[keep]

    if edge_attr is not None:
        edge_attr = edge_attr[keep]
    return torch.stack([row, col], dim=0), edge_attr


def install():
    """Attach `topk` and `filter_adj` to torch_geometric.nn.pool.topk_pool
    if the installed PyG version doesn't already provide them."""
    import torch_geometric.nn.pool.topk_pool as topk_pool_mod
    if not hasattr(topk_pool_mod, "topk"):
        topk_pool_mod.topk = _topk
    if not hasattr(topk_pool_mod, "filter_adj"):
        topk_pool_mod.filter_adj = _filter_adj