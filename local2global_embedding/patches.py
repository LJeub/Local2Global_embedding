"""Dividing input data into overlapping patches"""
from random import choice
from collections.abc import Sequence

import torch
import torch_geometric as tg
import torch_scatter as ts
from tqdm.auto import tqdm

from local2global_embedding.network import TGraph, conductance
from local2global_embedding.sparsify import resistance_sparsify, relaxed_spanning_tree, edge_sampling_sparsify


class Partition(Sequence):
    def __init__(self, partition_tensor):
        self.num_parts = torch.max(partition_tensor) + 1
        self.nodes = torch.argsort(partition_tensor)
        self.part_index = torch.zeros(self.num_parts + 1, dtype=torch.long, device=partition_tensor.device)
        ts.scatter(torch.ones(1, dtype=torch.long, device=partition_tensor.device).expand_as(partition_tensor),
                   partition_tensor, out=self.part_index[1:])
        self.part_index.cumsum_(0)

    def __getitem__(self, item):
        return self.nodes[self.part_index[item]:self.part_index[item+1]]

    def __len__(self):
        return self.num_parts


def geodesic_expand_overlap(subgraph: TGraph, source_nodes, min_overlap, target_overlap, reseed_samples=10):
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
    subgraph = subgraph.to(TGraph)
    target_overlap = int(target_overlap)
    if subgraph.num_nodes - len(source_nodes) < min_overlap:
        print(f"Minimum overlap {min_overlap} > other nodes {subgraph.num_nodes - len(source_nodes)}")
    mask = torch.ones(subgraph.num_nodes, dtype=torch.bool, device=subgraph.device)
    mask[source_nodes] = False
    new_nodes = torch.unique(torch.cat([subgraph.adj(node) for node in source_nodes]))
    new_nodes = new_nodes[mask[new_nodes]]
    mask[new_nodes] = False
    overlap = new_nodes
    if overlap.numel() > target_overlap:
        overlap = overlap[
            torch.multinomial(torch.ones(overlap.shape), target_overlap, replacement=False)]
    while overlap.numel() < min_overlap:
        new_nodes = torch.unique(torch.cat([subgraph.adj(node) for node in new_nodes]))
        new_nodes = new_nodes[mask[new_nodes]]
        if not new_nodes.numel():
            # no more connected nodes to add so add some remaining nodes by random sampling
            new_nodes = torch.nonzero(mask).flatten()
            if new_nodes.numel() > reseed_samples:
                new_nodes = new_nodes[
                    torch.multinomial(torch.ones(new_nodes.shape), reseed_samples, replacement=False)]
        if overlap.numel() + new_nodes.numel() > target_overlap:
            new_nodes = new_nodes[
                torch.multinomial(torch.ones(new_nodes.shape), target_overlap - overlap.numel(), replacement=False)]
        if not new_nodes.numel():
            raise RuntimeError("Could not reach minimum overlap.")
        mask[new_nodes] = False
        overlap = torch.cat((overlap, new_nodes))

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
    parts = Partition(partition_tensor)
    part_degs = torch.tensor([graph.degree[p].sum() for p in parts], device=graph.device)
    sizes = torch.tensor([p.numel() for p in parts], dtype=torch.long)
    smallest_id = torch.argmin(sizes)
    while sizes[smallest_id] < min_size:
        out_neighbour_fraction = torch.zeros(num_parts, device=graph.device)
        p = parts[smallest_id]
        for node in p:
            other = partition_tensor[graph.adj(node)]
            out_neighbour_fraction.scatter_add_(0, other, torch.ones(1).expand(other.shape, device=graph.device))
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


def create_overlapping_patches(graph: TGraph, partition_tensor: torch.LongTensor, patch_graph: TGraph, min_overlap,
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
    partition_tensor = partition_tensor.to(graph.device)
    parts = Partition(partition_tensor)
    patches = list(parts)
    print('enlarging patch overlaps')
    for (i, j) in tqdm(patch_graph.edges()):
        part_i = parts[i]
        part_j = parts[j]
        nodes = torch.cat((part_i, part_j))
        subgraph = graph.subgraph(nodes, keep_x=False, keep_y=False)
        patches[i] = torch.cat((patches[i],
                                nodes[geodesic_expand_overlap(
                                          subgraph,
                                          source_nodes=torch.arange(len(part_i), device=subgraph.device),
                                          min_overlap=min_overlap / 2,
                                          target_overlap=target_overlap / 2)]))
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

    partition_tensor = merge_small_clusters(graph, partition_tensor, min_patch_size)
    if verbose:
        print(f"number of patches: {partition_tensor.max().item() + 1}")
    pg = graph.partition_graph(partition_tensor).to(TGraph)
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
        pg = resistance_sparsify(pg, target_mean_degree=target_patch_degree)
    elif sparsify_method == 'rmst':
        pg = relaxed_spanning_tree(pg, maximise=True, gamma=gamma)
    elif sparsify_method == 'sample':
        pg = edge_sampling_sparsify(pg, target_patch_degree)
    elif sparsify_method == 'none':
        pass
    else:
        raise ValueError(
            f"Unknown sparsify method '{sparsify_method}', should be one of 'resistance', 'rmst', or 'none'.")

    if verbose:
        print(f"average patch degree: {pg.num_edges / pg.num_nodes}")

    patches = create_overlapping_patches(graph, partition_tensor, pg, min_overlap, target_overlap)
    patch_data = (graph.subgraph(patch, relabel=False).to(TGraph) for patch in patches)
    return patch_data, pg
