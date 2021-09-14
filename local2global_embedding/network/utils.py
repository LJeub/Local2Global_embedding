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
import numpy as np
import numba
from numba.experimental import jitclass

from local2global_embedding.network import NPGraph, TGraph
from .graph import Graph


@jitclass
class UnionFind:
    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.

      Union-find data structure. Based on Josiah Carlson's code,
      https://code.activestate.com/recipes/215912/
      with significant additional changes by D. Eppstein.
      http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py

    """
    parents: numba.int64[:]
    weights: numba.int64[:]

    def __init__(self, size):
        """Create a new empty union-find structure.

        If *elements* is an iterable, this structure will be initialized
        with the discrete partition on the given set of elements.

        """
        self.parents = np.arange(size, dtype=np.int64)
        self.weights = np.ones(size, dtype=np.int64)

    def find(self, i):
        """Find and return the name of the set containing the object."""

        # find path of objects leading to the root
        path = [i]
        root = self.parents[i]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def union(self, i, j):
        """Find the sets containing the objects and merge them all."""
        # Find the heaviest root according to its weight.
        roots = (self.find(i), self.find(j))
        if self.weights[roots[0]] < self.weights[roots[1]]:
            # heaviest root first
            roots = roots[::-1]

        self.weights[roots[0]] += self.weights[roots[1]]
        self.parents[roots[1]] = roots[0]


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


def spanning_tree_mask(graph: Graph, maximise=False):
    """Return an edge mask for minimum or maximum spanning tree edges.

    Args:
        graph: input graph
        maximise: if ``True``, find maximum spanning tree (default: ``False``)
    """

    convert_to_tensor = isinstance(graph, TGraph)
    graph = graph.to(NPGraph)

    # find positions of reverse edges
    if graph.undir:
        reverse_edge_index = np.argsort(graph.edge_index[1]*graph.num_nodes+graph.edge_index[0])
        edges = graph.edge_index[:, graph.edge_index[0] < graph.edge_index[1]]
        weights = graph.weights[graph.edge_index[0] < graph.edge_index[1]]
        reverse_edge_index = reverse_edge_index[graph.edge_index[0] < graph.edge_index[1]]
    else:
        edges = graph.edge_index
        weights = graph.weights
        reverse_edge_index = None

    index = np.argsort(weights)
    if maximise:
        index = index[::-1]

    edge_mask = np.zeros(graph.num_edges, dtype=np.bool)
    edge_mask = _spanning_tree_mask(edge_mask, edges, index, graph.num_nodes, reverse_edge_index)
    if convert_to_tensor:
        edge_mask = torch.as_tensor(edge_mask)
    return edge_mask


@numba.njit
def _spanning_tree_mask(edge_mask, edges, index, num_nodes, reverse_edge_index):
    subtrees = UnionFind(num_nodes)
    for it in range(len(index)):
        i = index[it]
        u = edges[0, i]
        v = edges[1, i]
        if subtrees.find(u) != subtrees.find(v):
            edge_mask[i] = True
            if reverse_edge_index is not None:
                edge_mask[reverse_edge_index[i]] = True
            subtrees.union(u, v)
    return edge_mask
