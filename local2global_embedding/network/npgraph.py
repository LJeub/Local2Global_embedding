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

import networkx as nx
import numpy as np
import torch
import numba

from .graph import Graph
from local2global_embedding import progress


@numba.njit
def _memmap_degree(edge_index, num_nodes):
    degree = np.zeros(num_nodes, dtype=np.int64)
    with numba.objmode:
        print('computing degrees')
        progress.reset_progress(edge_index.shape[1])
    for it, source in enumerate(edge_index[0]):
        degree[source] += 1
        if it % 1000000 == 0 and it > 0:
            with numba.objmode:
                progress.update_progress(1000000)
    with numba.objmode:
        progress.close_progress()
    return degree


class NPGraph(Graph):
    """
    numpy backed graph class with support for memmapped edge_index
    """
    @staticmethod
    def _convert_input(input):
        if input is None:
            return input
        else:
            return np.asanyarray(input)

    @classmethod
    def load(cls, folder, mmap_mode=None):
        folder = Path(folder)
        kwargs = {}

        kwargs['edge_index'] = np.load(folder / 'edge_index.npy', mmap_mode=mmap_mode)

        attr_file = folder / 'edge_attr.npy'
        if attr_file.is_file():
            kwargs['edge_attr'] = np.load(attr_file, mmap_mode=mmap_mode)

        info_file = folder / 'info.json'
        if info_file.is_file():
            with open(info_file) as f:
                info = json.load(f)
            kwargs.update(info)

        feat_file = folder / 'node_feat.npy'
        if feat_file.is_file():
            kwargs['x'] = np.load(feat_file, mmap_mode=mmap_mode)

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
        else:
            self.num_nodes = self.num_nodes

        if ensure_sorted:
            if isinstance(self.edge_index, np.memmap):
                raise NotImplementedError("Sorting for memmapped arrays not yet implemented")
            else:
                index = np.argsort(self.edge_index[0]*self.num_nodes + self.edge_index[1])
                self.edge_index = self.edge_index[:, index]
                if self.edge_attr is not None:
                    self.edge_attr = self.edge_attr[index]

        if self.adj_index is None:
            if isinstance(self.edge_index, np.memmap):
                self.degree = _memmap_degree(self.edge_index, self.num_nodes)
            else:
                self.degree = np.zeros(self.num_nodes, dtype=np.int64)
                np.add.at(self.degree, self.edge_index[0], 1)
            self.adj_index = np.zeros(self.num_nodes + 1, dtype=np.int64)  #: adjacency index such that edges starting at node ``i`` are given by ``edge_index[:, adj_index[i]:adj_index[i+1]]``
            self.degree.cumsum(out=self.adj_index[1:])
        else:
            self.degree = self.adj_index[1:]-self.adj_index[:-1]

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
        return [(e[0], e[1], w[0] if w.size > 1 else w)
                for e, w in zip(self.edge_index.T, self.weights)]

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

    def subgraph(self, nodes: torch.Tensor):
        """
        find induced subgraph for a set of nodes

        Args:
            nodes: node indeces

        Returns:
            subgraph

        """
        index = np.concatenate([np.arange(self.adj_index[node], self.adj_index[node + 1], dtype=np.int64) for node in nodes])
        node_mask = np.zeros(self.num_nodes, dtype=np.bool)
        node_mask[nodes] = True
        node_ids = np.zeros(self.num_nodes, dtype=np.int64)
        node_ids[nodes] = np.arange(len(nodes))
        index = index[node_mask[self.edge_index[1][index]]]
        edge_attr = self.edge_attr
        return self.__class__(edge_index=node_ids[self.edge_index[:, index]],
                              edge_attr=edge_attr[index] if edge_attr is not None else None,
                              num_nodes=len(nodes),
                              ensure_sorted=False,
                              undir=self.undir)

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
        append_pointer = 1
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

        component_id, inverse, component_size = np.unique(components, return_counts=True, return_inverse=True)
        new_id = np.argsort(component_size)[::-1]
        return new_id[inverse]

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
