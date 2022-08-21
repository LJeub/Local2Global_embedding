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

import json
from pathlib import Path
from tempfile import TemporaryFile
from random import randrange

import numpy as np
import torch
import numba
from numba.experimental import jitclass

from .graph import Graph
from local2global_embedding import progress


rng = np.random.default_rng()


spec = [
    ('edge_index', numba.int64[:, :]),
    ('adj_index', numba.int64[:]),
    ('degree', numba.int64[:]),

]


class NPGraph(Graph):
    """
    numpy backed graph class with support for memmapped edge_index
    """
    @staticmethod
    def _convert_input(input):
        if input is None:
            return input
        elif isinstance(input, torch.Tensor):
            return np.asanyarray(input.cpu())
        else:
            return np.asanyarray(input)

    @classmethod
    def load(cls, folder, mmap_edges=None, mmap_features=None):
        folder = Path(folder)
        kwargs = {}

        kwargs['edge_index'] = np.load(folder / 'edge_index.npy', mmap_mode=mmap_edges)

        attr_file = folder / 'edge_attr.npy'
        if attr_file.is_file():
            kwargs['edge_attr'] = np.load(attr_file, mmap_mode=mmap_edges)

        info_file = folder / 'info.json'
        if info_file.is_file():
            with open(info_file) as f:
                info = json.load(f)
            kwargs.update(info)

        feat_file = folder / 'node_feat.npy'
        if feat_file.is_file():
            kwargs['x'] = np.load(feat_file, mmap_mode=mmap_features)

        label_file = folder / 'node_label.npy'
        if label_file.is_file():
            kwargs['y'] = np.load(label_file)

        index_file = folder / 'adj_index.npy'
        if index_file.is_file():
            kwargs['adj_index'] = np.load(index_file)

        return cls(**kwargs)

    def save(self, folder):
        folder = Path(folder)
        np.save(folder / 'edge_index.npy', self.edge_index)

        if self.weighted:
            np.save(folder / 'edge_attr.npy', self.edge_attr)

        np.save(folder / 'adj_index.npy', self.adj_index)

        info = {'num_nodes': self.num_nodes, 'undir': self.undir}
        with open(folder / 'info.json', 'w') as f:
            json.dump(info, f)

        if self.y is not None:
            np.save(self.y, folder / 'node_label.npy')

        if self.x is not None:
            np.save(self.x, folder / 'node_feat.npy')

    def __init__(self, *args, ensure_sorted=False, **kwargs):
        super().__init__(*args, **kwargs)

        if self.num_nodes is None:
            self.num_nodes = np.max(self.edge_index) + 1

        if ensure_sorted:
            if isinstance(self.edge_index, np.memmap):
                raise NotImplementedError("Sorting for memmapped arrays not yet implemented")
            else:
                index = np.argsort(self.edge_index[0]*self.num_nodes + self.edge_index[1])
                self.edge_index = self.edge_index[:, index]
                if self.edge_attr is not None:
                    self.edge_attr = self.edge_attr[index]
        self._jitgraph = JitGraph(self.edge_index, self.num_nodes, self.adj_index, None)
        self.adj_index = self._jitgraph.adj_index
        self.degree = self._jitgraph.degree
        self.num_nodes = self._jitgraph.num_nodes

        if self.weighted:
            self.weights = self.edge_attr
            self.strength = np.zeros(self.num_nodes)  #: tensor of node strength
            np.add.at(self.strength, self.edge_index[0], self.weights)
        else:
            self.weights = np.broadcast_to(np.ones(1), (self.num_edges,))  # use expand to avoid actually allocating large array
            self.strength = self.degree
        self.device = 'cpu'

        if self.undir is None:
            if isinstance(self.edge_index, np.memmap):
                raise NotImplementedError("Checking directedness for memmapped arrays not yet implemented")
            else:
                index = np.argsort(self.edge_index[1]*self.num_nodes + self.edge_index[0])
                edge_reverse = self.edge_index[::-1, index]
                self.undir = np.array_equal(self.edge_index, edge_reverse)
                if self.weighted:
                    self.undir = self.undir and np.array_equal(self.weights, self.weights[index])

    def edges(self):
        """
        return list of edges where each edge is a tuple ``(source, target)``
        """
        return ((e[0], e[1]) for e in self.edge_index.T)

    def edges_weighted(self):
        """
        return list of edges where each edge is a tuple ``(source, target, weight)``
        """
        return ((e[0], e[1], w[0] if w.size > 1 else w)
                for e, w in zip(self.edge_index.T, self.weights))

    def is_edge(self, source, target):
        return self._jitgraph.is_edge(source, target)

    def neighbourhood(self, nodes, hops: int = 1):
        """
        find the neighbourhood of a set of source nodes

        note that the neighbourhood includes the source nodes themselves

        Args:
            nodes: indices of source nodes
            hops: number of hops for neighbourhood

        Returns:
            neighbourhood

        """
        explore = np.ones(self.num_nodes, dtype=np.bool)
        explore[nodes] = False
        all_nodes = nodes
        new_nodes = nodes
        for _ in range(hops):
            new_nodes = np.concatenate([self.adj(node) for node in new_nodes])
            new_nodes = np.unique(new_nodes[explore[new_nodes]])
            explore[new_nodes] = False
            all_nodes = np.concatenate((all_nodes, new_nodes))
        return all_nodes

    def subgraph(self, nodes: torch.Tensor, relabel=False, keep_x=True, keep_y=True):
        """
        find induced subgraph for a set of nodes

        Args:
            nodes: node indeces

        Returns:
            subgraph

        """
        nodes = np.asanyarray(nodes)
        edge_index, index = self._jitgraph.subgraph_edges(nodes)
        edge_attr = self.edge_attr
        if relabel:
            node_labels = None
        else:
            node_labels = [self.nodes[n] for n in nodes]
        if self.x is not None and keep_x:
            x = self.x[nodes]
        else:
            x = None
        if self.y is not None and keep_y:
            y = self.y[nodes]
        else:
            y = None
        return self.__class__(edge_index=edge_index,
                              edge_attr=edge_attr[index] if edge_attr is not None else None,
                              num_nodes=len(nodes),
                              ensure_sorted=False,
                              undir=self.undir,
                              nodes=node_labels,
                              x=x,
                              y=y)

    def connected_component_ids(self):
        """
        return nodes in breadth-first-search order

        Args:
            start: index of starting node (default: 0)

        Returns:
            tensor of node indeces

        """
        return self._jitgraph.connected_component_ids()

    def nodes_in_lcc(self):
        """List all nodes in the largest connected component"""
        return np.flatnonzero(self.connected_component_ids() == 0)

    def bfs_order(self, start=0):
        """
        return nodes in breadth-first-search order

        Args:
            start: index of starting node (default: 0)

        Returns:
            tensor of node indeces

        """
        bfs_list = np.full((self.num_nodes,), -1, dtype=np.int64)
        not_visited = np.ones(self.num_nodes, dtype=np.int64)
        bfs_list[0] = start
        not_visited[start] = False
        append_pointer = 1
        i = 0
        restart = 0
        while append_pointer < self.num_nodes:
            node = bfs_list[i]
            if node < 0:
                for node in range(restart, self.num_nodes):
                    if not_visited[node]:
                        break
                restart = node
                bfs_list[i] = node
                not_visited[node] = False
                append_pointer += 1
            i += 1
            new_nodes = self.adj(node)
            new_nodes = new_nodes[not_visited[new_nodes]]
            number_new_nodes = len(new_nodes)
            not_visited[new_nodes] = False
            bfs_list[append_pointer:append_pointer+number_new_nodes] = new_nodes
            append_pointer += number_new_nodes
        return bfs_list

    def partition_graph(self, partition, self_loops=True):
        partition = np.asanyarray(partition)
        partition_edges, weights = self._jitgraph.partition_graph_edges(partition, self_loops)
        return self.__class__(edge_index=partition_edges, edge_attr=weights, undir=self.undir)

    def sample_negative_edges(self, num_samples):
        return self._jitgraph.sample_negative_edges(num_samples)

    def sample_positive_edges(self, num_samples):
        index = rng.integers(self.num_edges, size=(num_samples,))
        return self.edge_index[:, index]


@numba.njit
def _subgraph_edges(edge_index, adj_index, degs, num_nodes, sources):
    max_edges = degs[sources].sum()
    subgraph_edge_index = np.empty((2, max_edges), dtype=np.int64)
    index = np.empty((max_edges,), dtype=np.int64)
    target_index = np.full((num_nodes,), -1, np.int64)
    target_index[sources] = np.arange(len(sources))
    count = 0

    for s in range(len(sources)):
        for i in range(adj_index[sources[s]], adj_index[sources[s]+1]):
            t = target_index[edge_index[1, i]]
            if t >= 0:
                subgraph_edge_index[0, count] = s
                subgraph_edge_index[1, count] = t
                index[count] = i
                count += 1
    return subgraph_edge_index[:, :count], index[:count]


@numba.njit
def _memmap_degree(edge_index, num_nodes):
    degree = np.zeros(num_nodes, dtype=np.int64)
    with numba.objmode:
        print('computing degrees')
        progress.reset(edge_index.shape[1])
    for it, source in enumerate(edge_index[0]):
        degree[source] += 1
        if it % 1000000 == 0 and it > 0:
            with numba.objmode:
                progress.update(1000000)
    with numba.objmode:
        progress.close()
    return degree


@jitclass(
    [
        ('edge_index', numba.int64[:, :]),
        ('adj_index', numba.int64[:]),
        ('degree', numba.int64[:]),
        ('num_nodes', numba.int64)
    ]
)
class JitGraph:
    def __init__(self, edge_index, num_nodes=None, adj_index=None, degree=None):
        if num_nodes is None:
            num_nodes_int = edge_index.max() + 1
        else:
            num_nodes_int = num_nodes

        if adj_index is None:
            adj_index_ar = np.zeros((num_nodes_int+1,), dtype=np.int64)
        else:
            adj_index_ar = adj_index

        if degree is None:
            if adj_index is None:
                degree = np.zeros((num_nodes_int,), dtype=np.int64)
                for s in edge_index[0]:
                    degree[s] += 1
                adj_index_ar[1:] = degree.cumsum()
            else:
                degree = adj_index_ar[1:]-adj_index_ar[:-1]

        self.edge_index = edge_index
        self.adj_index = adj_index_ar
        self.degree = degree
        self.num_nodes = num_nodes_int

    def is_edge(self, source, target):
        if source not in range(self.num_nodes) or target not in range(self.num_nodes):
            return False
        index = np.searchsorted(self.edge_index[1, self.adj_index[source]:self.adj_index[source + 1]], target)
        if index < self.degree[source] and self.edge_index[1, self.adj_index[source] + index] == target:
            return True
        else:
            return False

    def sample_negative_edges(self, num_samples):
        i = 0
        sampled_edges = np.empty((2, num_samples), dtype=np.int64)
        while i < num_samples:
            source = randrange(self.num_nodes)
            target = randrange(self.num_nodes)
            if not self.is_edge(source, target):
                sampled_edges[0, i] = source
                sampled_edges[1, i] = target
                i += 1
        return sampled_edges

    def adj(self, node):
        return self.edge_index[1][self.adj_index[node]:self.adj_index[node+1]]

    def neighbours(self, nodes):
        size = self.degree[nodes].sum()
        out = np.empty((size,), dtype=np.int64)
        it = 0
        for node in nodes:
            out[it:it+self.degree[node]] = self.adj(node)
            it += self.degree[node]
        return np.unique(out)

    def sample_positive_edges(self, num_samples):
        index = np.random.randint(self.num_edges, (num_samples,))
        return self.edge_index[:, index]

    def subgraph_edges(self, sources):
        max_edges = self.degree[sources].sum()
        subgraph_edge_index = np.empty((2, max_edges), dtype=np.int64)
        index = np.empty((max_edges,), dtype=np.int64)
        target_index = np.full((self.num_nodes,), -1, np.int64)
        target_index[sources] = np.arange(len(sources))
        count = 0

        for s in range(len(sources)):
            for ei in range(self.adj_index[sources[s]], self.adj_index[sources[s]+1]):
                t = target_index[self.edge_index[1][ei]]
                if t >= 0:
                    subgraph_edge_index[0, count] = s
                    subgraph_edge_index[1, count] = t
                    index[count] = ei
                    count += 1
        return subgraph_edge_index[:, :count], index[:count]

    def subgraph(self, sources):
        edge_index, _ = self.subgraph_edges(sources)
        return JitGraph(edge_index, len(sources), None, None)

    def partition_graph_edges(self, partition, self_loops):
        num_edges = self.num_edges
        with numba.objmode:
            print('finding partition edges')
            progress.reset(num_edges)
        num_clusters = partition.max() + 1
        edge_counts = np.zeros((num_clusters, num_clusters), dtype=np.int64)
        for i, (source, target) in enumerate(self.edge_index.T):
            source = partition[source]
            target = partition[target]
            if self_loops or (source != target):
                edge_counts[source, target] += 1
            if i % 1000000 == 0 and i > 0:
                with numba.objmode:
                    progress.update(1000000)
        with numba.objmode:
            progress.close()
        index = np.nonzero(edge_counts)
        partition_edges = np.vstack(index)
        weights = np.empty((len(index[0]),), dtype=np.int64)
        for it, (i, j) in enumerate(zip(*index)):
            weights[it] = edge_counts[i][j]
        return partition_edges, weights

    def partition_graph(self, partition, self_loops):
        edge_index, _ = self.partition_graph_edges(partition, self_loops)
        return JitGraph(edge_index, None, None, None)

    def connected_component_ids(self):
        """
                return nodes in breadth-first-search order

                Args:
                    start: index of starting node (default: 0)

                Returns:
                    tensor of node indeces

                """
        components = np.full((self.num_nodes,), -1, dtype=np.int64)
        not_visited = np.ones(self.num_nodes, dtype=np.bool)
        component_id = 0
        components[0] = component_id
        not_visited[0] = False
        bfs_list = [0]
        i = 0
        for _ in range(self.num_nodes):
            if bfs_list:
                node = bfs_list.pop()
            else:
                component_id += 1
                for i in range(i, self.num_nodes):
                    if not_visited[i]:
                        break
                node = i
                not_visited[node] = False
            components[node] = component_id
            new_nodes = self.adj(node)
            new_nodes = new_nodes[not_visited[new_nodes]]
            not_visited[new_nodes] = False
            bfs_list.extend(new_nodes)

        num_components = components.max()+1
        component_size = np.zeros((num_components,), dtype=np.int64)
        for i in components:
            component_size[i] += 1
        new_id = np.argsort(component_size)[::-1]
        inverse = np.empty_like(new_id)
        inverse[new_id] = np.arange(num_components)
        return inverse[components]

    def nodes_in_lcc(self):
        """List all nodes in the largest connected component"""
        return np.flatnonzero(self.connected_component_ids() == 0)

    @property
    def num_edges(self):
        return self.edge_index.shape[1]
