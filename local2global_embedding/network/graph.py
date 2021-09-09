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
from typing import Sequence, Collection, Iterable
import networkx as nx
from abc import ABC, abstractmethod


class Graph:
    """
    numpy backed graph class with support for memmapped edge_index
    """
    weights: Sequence
    degree: Sequence
    device = 'cpu'

    @staticmethod
    def _convert_input(input):
        return input

    @classmethod
    def from_tg(cls, data):
        return cls(edge_index=data.edge_index,
                   edge_attr=data.edge_attr,
                   x=data.x,
                   y=data.y,
                   num_nodes=data.num_nodes)

    @abstractmethod
    def __init__(self, edge_index, edge_attr=None, x=None, y=None, num_nodes=None, adj_index=None,
                 ensure_sorted=False, undir=None):
        """
        Initialise graph

        Args:
            edge_index: edge index such that ``edge_index[0]`` lists the source and ``edge_index[1]`` the target node for each edge
            edge_attr: optionally provide edge weights
            num_nodes: specify number of nodes (default: ``max(edge_index)+1``)
            ensure_sorted: if ``False``, assume that the ``edge_index`` input is already sorted
            undir: boolean indicating if graph is directed. If not provided, the ``edge_index`` is checked to determine this value.
        """
        self.edge_index = self._convert_input(edge_index)
        self.edge_attr = self._convert_input(edge_attr)
        self.x = self._convert_input(x)
        self.y = self._convert_input(y)
        self.num_nodes = num_nodes
        if self.num_nodes is not None:
            self.num_nodes = int(num_nodes)
        self.undir = undir
        self.adj_index = self._convert_input(adj_index)

    @property
    def weighted(self):
        """boolean indicating if graph is weighted"""
        return self.edge_attr is not None

    @property
    def num_edges(self):
        return self.edge_index.shape[1]

    def adj(self, node: int):
        """
        list neighbours of node

        Args:
            node: source node

        Returns:
            neighbours

        """
        return self.edge_index[1][self.adj_index[node]:self.adj_index[node + 1]]

    def adj_weighted(self, node: int):
        """
        list neighbours of node and corresponding edge weight
        Args:
            node: source node

        Returns:
            neighbours, weights

        """
        return self.adj(node), self.weights[self.adj_index[node]:self.adj_index[node + 1]]

    @abstractmethod
    def edges(self):
        """
        iterator over edges
        """
        raise NotImplementedError

    @abstractmethod
    def edges_weighted(self):
        """
        iterator over weighted edges where each edge is a tuple ``(source, target, weight)``
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def subgraph(self, nodes: Iterable):
        """
        find induced subgraph for a set of nodes

        Args:
            nodes: node indeces

        Returns:
            subgraph

        """
        raise NotImplementedError

    @abstractmethod
    def connected_component_ids(self):
        """
        return connected component ids where ids are sorted in decreasing order by component size

        Returns:
            Sequence of node indeces

        """
        raise NotImplementedError

    def nodes_in_lcc(self):
        """Iterator over nodes in the largest connected component"""
        return (i for i, c in enumerate(self.connected_component_ids()) if c == 0)

    def lcc(self):
        return self.subgraph(self.nodes_in_lcc())

    def to_networkx(self):
        """convert graph to NetworkX format"""
        if self.undir:
            nxgraph = nx.Graph()
        else:
            nxgraph = nx.DiGraph()
        nxgraph.add_nodes_from(range(self.num_nodes))
        if self.weighted:
            nxgraph.add_weighted_edges_from(self.edges_weighted())
        else:
            nxgraph.add_edges_from(self.edges())
        return nxgraph

    def to(self, graph_cls):
        if self.__class__ is graph_cls:
            return self
        else:
            return graph_cls(edge_index=self.edge_index,
                             edge_attr=self.edge_attr,
                             x=self.x,
                             y=self.y,
                             num_nodes=self.num_nodes,
                             adj_index=self.adj_index,
                             ensure_sorted=False,
                             undir=self.undir)

    @abstractmethod
    def bfs_order(self, start=0):
        """
        return nodes in breadth-first-search order

        Args:
            start: index of starting node (default: 0)

        Returns:
           Sequence of node indeces

        """
        raise NotImplementedError
