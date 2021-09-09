"""Graph data handling"""

#  Copyright (c) 2021. Lucas G. S. Jeub
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import torch
from networkx.utils import UnionFind

from local2global_embedding.network.tgraph import TGraph
from .graph import Graph


def conductance(graph: Graph, source, target=None):
    """
    compute conductance between source and target nodes

    Args:
        graph: input graph
        source: set of source nodes
        target: set of target nodes (if ``target=None``, consider all nodes that are not in ``source`` as target)

    Returns:
        conductance

    """
    if target is None:
        target_mask = torch.ones(graph.num_nodes, dtype=torch.bool, device=graph.device)
        target_mask[source] = False
    else:
        target_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        target_mask[target] = True
    out = torch.cat([graph.adj(node) for node in source])
    cond = torch.sum(target_mask[out]).float()
    s_deg = graph.degree[source].sum()
    t_deg = graph.num_edges-s_deg if target is None else graph.degree[target].sum()
    cond /= torch.minimum(s_deg, t_deg)
    return cond


def spanning_tree(graph: TGraph, maximise=False):
    """Implements Kruskal's algorithm for finding minimum or maximum spanning tree.

    Args:
        graph: input graph
        maximise: if ``True``, find maximum spanning tree (default: ``False``)

    Returns:
        spanning tree
    """
    edge_mask = spanning_tree_mask(graph, maximise)

    edge_index = graph.edge_index[:, edge_mask]
    if graph.edge_attr is not None:
        weights = graph.edge_attr[edge_mask]
    else:
        weights = None
    return TGraph(edge_index=edge_index, edge_attr=weights, num_nodes=graph.num_nodes, ensure_sorted=False)


def spanning_tree_mask(graph: TGraph, maximise=False):
    """Return an edge mask for minimum or maximum spanning tree edges.

    Args:
        graph: input graph
        maximise: if ``True``, find maximum spanning tree (default: ``False``)
    """

    # find positions of reverse edges
    if graph.undir:
        reverse_edge_index = torch.argsort(graph.edge_index[1]*graph.num_nodes+graph.edge_index[0])

    if graph.edge_attr is not None:
        index = torch.argsort(graph.edge_attr, descending=maximise)
    else:
        index = torch.arange(graph.num_edges)
    edges = graph.edge_index
    subtrees = UnionFind()
    edge_mask = torch.zeros(graph.num_edges, dtype=torch.bool, device=graph.device)
    for i in index:
        u = edges[0, i].item()
        v = edges[1, i].item()
        if subtrees[u] != subtrees[v]:
            edge_mask[i] = True
            if graph.undir:
                edge_mask[reverse_edge_index[i]] = True
            subtrees.union(u, v)
    return edge_mask
