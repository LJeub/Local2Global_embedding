"""Dividing input data into overlapping patches"""
from random import choice

import torch
import numpy as np
from tqdm.auto import tqdm

import numba

from local2global_embedding.clustering import Partition
from local2global_embedding.network import TGraph, NPGraph
from local2global_embedding.network.npgraph import JitGraph
from local2global_embedding.sparsify import resistance_sparsify, relaxed_spanning_tree, edge_sampling_sparsify, hierarchical_sparsify


@numba.njit
def geodesic_expand_overlap(subgraph, seed_mask, min_overlap, target_overlap, reseed_samples=10):
    """
    expand patch

    Args:
        subgraph: graph containing patch nodes and all target nodes for potential expansion
        source_nodes: index of source nodes (initial starting nodes for expansion)
        min_overlap: minimum overlap before stopping expansion
        target_overlap: maximum overlap (if expansion step results in more overlap, the nodes
                        added are sampled at random)

    Returns:
        index tensor of new nodes to add to patch
    """
    if subgraph.num_nodes < min_overlap:
        raise RuntimeError("Minimum overlap > number of nodes")
    mask = ~seed_mask
    new_nodes = np.flatnonzero(seed_mask)
    overlap = new_nodes
    if overlap.size > target_overlap:
        overlap = np.random.choice(overlap, target_overlap, replace=False)
    while overlap.size < min_overlap:
        new_nodes = subgraph.neighbours(new_nodes)
        new_nodes = new_nodes[mask[new_nodes]]
        if not new_nodes.size:
            # no more connected nodes to add so add some remaining nodes by random sampling
            new_nodes = np.flatnonzero(mask)
            if new_nodes.size > reseed_samples:
                new_nodes = np.random.choice(new_nodes, reseed_samples, replace=False)
        if overlap.size + new_nodes.size > target_overlap:
            new_nodes = np.random.choice(new_nodes, target_overlap - overlap.size, replace=False)
        if not new_nodes.size:
            raise RuntimeError("Could not reach minimum overlap.")
        mask[new_nodes] = False
        overlap = np.concatenate((overlap, new_nodes))
    return overlap


def merge_small_clusters(graph: TGraph, partition_tensor: torch.LongTensor, min_size):
    """
    Merge small clusters with adjacent clusters such that all clusters satisfy a minimum size constraint.

    This function iteratively merges the smallest cluster with its neighbouring cluster with the
    maximum normalized cut.

    Args:
        graph: Input graph
        partition_tensor: input partition vector mapping nodes to clusters
        min_size: desired minimum size of clusters

    Returns:
        new partition tensor where small clusters are merged.
    """
    parts = [torch.as_tensor(p, device=graph.device) for p in Partition(partition_tensor)]
    num_parts = len(parts)
    part_degs = torch.tensor([graph.degree[p].sum() for p in parts], device=graph.device)
    sizes = torch.tensor([len(p) for p in parts], dtype=torch.long)
    smallest_id = torch.argmin(sizes)
    while sizes[smallest_id] < min_size:
        out_neighbour_fraction = torch.zeros(num_parts, device=graph.device)
        p = parts[smallest_id]
        for node in p:
            other = partition_tensor[graph.adj(node)]
            out_neighbour_fraction.scatter_add_(0, other, torch.ones(1, device=graph.device).expand(other.shape))
        if out_neighbour_fraction.sum() == 0:
            merge = torch.argsort(sizes)[1]
        else:
            out_neighbour_fraction /= part_degs  # encourage merging with smaller clusters
            out_neighbour_fraction[smallest_id] = 0
            merge = torch.argmax(out_neighbour_fraction)
        if merge > smallest_id:
            new_id = smallest_id
            other = merge
        else:
            new_id = merge
            other = smallest_id

        partition_tensor[parts[other]] = new_id
        sizes[new_id] += sizes[other]
        part_degs[new_id] += part_degs[other]
        parts[new_id] = torch.cat((parts[new_id], parts[other]))
        if other < num_parts - 1:
            partition_tensor[parts[-1]] = other
            sizes[other] = sizes[-1]
            part_degs[other] = part_degs[-1]
            parts[other] = parts[-1]
        num_parts = num_parts - 1
        sizes = sizes[:num_parts]
        part_degs = part_degs[:num_parts]
        parts = parts[:num_parts]
        smallest_id = torch.argmin(sizes)
    return partition_tensor


def create_overlapping_patches(graph, partition_tensor: torch.LongTensor, patch_graph, min_overlap,
                               target_overlap):
    """
    Create overlapping patches from a hard partition of an input graph

    Args:
        graph: input graph
        partition_tensor: partition of input graph
        patch_graph: graph where nodes are clusters of partition and an edge indicates that the corresponding
                     patches in the output should have at least ``min_overlap`` nodes in common
        min_overlap: minimum overlap for connected patches
        target_overlap: maximum overlap during expansion for an edge (additional overlap may
                        result from expansion of other edges)

    Returns:
        list of node-index tensors for patches

    """
    if isinstance(partition_tensor, torch.Tensor):
        partition_tensor = partition_tensor.cpu()
    graph = graph.to(NPGraph)._jitgraph
    patch_graph = patch_graph.to(NPGraph)._jitgraph
    parts = Partition(partition_tensor)
    partition_tensor = partition_tensor.numpy()
    patches = numba.typed.List(np.asanyarray(p) for p in parts)
    for i in tqdm(range(patch_graph.num_nodes), desc='enlarging patch overlaps'):
        part_i = parts[i].numpy()
        part_i.sort()
        patches = _patch_overlaps(i, part_i, partition_tensor, patches, graph, patch_graph, int(min_overlap / 2), int(target_overlap / 2))

    return patches


@numba.njit
def _patch_overlaps(i, part, partition, patches, graph, patch_graph, min_overlap, target_overlap):
    max_edges = graph.degree[part].sum()
    edge_index = np.empty((2, max_edges), dtype=np.int64)
    adj_index = np.zeros((len(part)+1,), dtype=np.int64)
    part_index = np.full((graph.num_nodes,), -1, dtype=np.int64)
    part_index[part] = np.arange(len(part))

    patch_index = np.full((patch_graph.num_nodes,), -1, dtype=np.int64)
    patch_index[patch_graph.adj(i)] = np.arange(patch_graph.degree[i])
    source_mask = np.zeros((part.size, patch_graph.degree[i]), dtype=np.bool_)  # track source nodes for different patches
    edge_count = 0
    for index in range(len(part)):
        targets = graph.adj(part[index])
        for t in part_index[targets]:
            if t >= 0:
                edge_index[0, edge_count] = index
                edge_index[1, edge_count] = t
                edge_count += 1
        adj_index[index+1] = edge_count
        pi = patch_index[partition[targets]]
        pi = pi[pi >= 0]
        source_mask[index][pi] = True
    edge_index = edge_index[:, :edge_count]
    subgraph = JitGraph(edge_index, len(part), adj_index, None)

    for it, j in enumerate(patch_graph.adj(i)):
        patches[j] = np.concatenate((patches[j],
                                     part[geodesic_expand_overlap(
                                         subgraph,
                                         seed_mask=source_mask[:, it],
                                         min_overlap=min_overlap,
                                         target_overlap=target_overlap)]))
    return patches


def create_patch_data(graph: TGraph, partition_tensor, min_overlap, target_overlap,
                      min_patch_size=None, sparsify_method='resistance', target_patch_degree=4, gamma=0, verbose=False):
    """
    Divide data into overlapping patches

    Args:
        graph: input data
        partition_tensor: starting partition for creating patches
        min_overlap: minimum patch overlap for connected patches
        target_overlap: maximum patch overlap during expansion of an edge of the patch graph
        min_patch_size: minimum size of patches
        sparsify_method: method for sparsifying patch graph (one of ``'resistance'``, ``'rmst'``, ``'none'``)
        target_patch_degree: target patch degree for ``sparsify_method='resistance'``
        gamma: ``gamma`` value for use with ``sparsify_method='rmst'``
        verbose: if true, print some info about created patches

    Returns:
        list of patch data, patch graph

    """
    if min_patch_size is None:
        min_patch_size = min_overlap

    if isinstance(partition_tensor, list):
        partition_tensor_0 = partition_tensor[0]
    else:
        partition_tensor = merge_small_clusters(graph, partition_tensor, min_patch_size)
        partition_tensor_0 = partition_tensor

    if verbose:
        print(f"number of patches: {partition_tensor_0.max().item() + 1}")
    pg = graph.partition_graph(partition_tensor_0).to(TGraph)
    components = pg.connected_component_ids()
    num_components = components.max()+1
    if num_components > 1:
        # connect all components
        edges = torch.empty((2, num_components*(num_components-1)/2), dtype=torch.long)
        comp_lists = [[] for _ in range(num_components)]
        for i, c in enumerate(components):
            comp_lists[c].append(i)
        i = 0
        for c1 in range(num_components):
            for c2 in range(c1+1, num_components):
                p1 = choice(comp_lists[c1])
                p2 = choice(comp_lists[c2])
                edges[:, i] = (p1, p2)
                i += 1

        edge_index = torch.cat((pg.edge_index, edges, edges[::-1, :]))
        weights = torch.cat((pg.edge_attr, torch.ones(2*edges.shape[1], dtype=torch.long)))
        pg = TGraph(edge_index=edge_index, edge_attr=weights, ensure_sorted=True, num_nodes=pg.num_nodes,
                    undir=pg.undir)

    if sparsify_method == 'resistance':
        if isinstance(partition_tensor, list):
            pg = hierarchical_sparsify(pg, partition_tensor[1:], target_patch_degree, sparsifier=resistance_sparsify)
        else:
            pg = resistance_sparsify(pg, target_mean_degree=target_patch_degree)
    elif sparsify_method == 'rmst':
        pg = relaxed_spanning_tree(pg, maximise=True, gamma=gamma)
    elif sparsify_method == 'sample':
        if isinstance(partition_tensor, list):
            pg = hierarchical_sparsify(pg, partition_tensor[1:], target_patch_degree, sparsifier=edge_sampling_sparsify)
        else:
            pg = edge_sampling_sparsify(pg, target_patch_degree)
    elif sparsify_method == 'none':
        pass
    else:
        raise ValueError(
            f"Unknown sparsify method '{sparsify_method}', should be one of 'resistance', 'rmst', or 'none'.")

    if verbose:
        print(f"average patch degree: {pg.num_edges / pg.num_nodes}")

    patches = create_overlapping_patches(graph, partition_tensor_0, pg, min_overlap, target_overlap)
    return patches, pg


def rolling_window_edges(n_patches, w):
    """
    Generate patch edges for a rolling window

    Args:
        n_patches: Number of patches
        w: window width (patches connected to the w nearest neighbours on either side)

    """
    for i in range(n_patches):
        for j in range(max(i-w, 0), min(i+w+1, n_patches)):
            if i != j:
                yield i, j
