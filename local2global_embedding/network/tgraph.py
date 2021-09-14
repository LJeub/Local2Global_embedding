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
import typing as _t

import networkx as nx
import torch
import torch_scatter as ts
import torch_geometric as tg

from .graph import Graph


class TGraph(Graph):
    """Wrapper class for pytorch-geometric edge_index providing fast adjacency look-up."""
    @staticmethod
    def _convert_input(input):
        if input is None:
            return None
        else:
            return torch.as_tensor(input)

    def __init__(self, *args, ensure_sorted=False, **kwargs):
        super().__init__(*args, **kwargs)

        if self.num_nodes is None:
            self.num_nodes = int(torch.max(self.edge_index)+1)  #: number of nodes

        if ensure_sorted:
            index = torch.argsort(self.edge_index[0]*self.num_nodes+self.edge_index[1])
            self.edge_index = self.edge_index[:, index]
            if self.edge_attr is not None:
                self.edge_attr = self.edge_attr[index]

        if self.adj_index is None:
            self.degree = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)  #: tensor of node degrees
            self.degree.index_add_(0, self.edge_index[0],
                                   torch.ones(1, dtype=torch.long, device=self.device).expand(self.num_edges))  # use expand to avoid actually allocating large array
            self.adj_index = torch.zeros(self.num_nodes + 1, dtype=torch.long)  #: adjacency index such that edges starting at node ``i`` are given by ``edge_index[:, adj_index[i]:adj_index[i+1]]``
            self.adj_index[1:] = torch.cumsum(self.degree, 0)
        else:
            self.degree = self.adj_index[1:] - self.adj_index[:-1]

        if self.weighted:
            self.weights = self.edge_attr
            self.strength = torch.zeros(self.num_nodes, device=self.device, dtype=self.weights.dtype)  #: tensor of node strength
            self.strength.index_add_(0, self.edge_index[0], self.weights)
        else:
            # use expand to avoid actually allocating large array
            self.weights = torch.ones(1, device=self.device).expand(self.num_edges)
            self.strength = self.degree

        if self.undir is None:
            index = torch.argsort(self.edge_index[1]*self.num_nodes+self.edge_index[0])
            self.undir = torch.equal(self.edge_index, self.edge_index[:, index].flip((0,)))
            if self.weighted:
                self.undir = self.undir and torch.equal(self.weights, self.weights[index])

    @property
    def device(self):
        """device holding graph data"""
        return self.edge_index.device

    def edges(self):
        """
        return list of edges where each edge is a tuple ``(source, target)``
        """
        return ((self.edge_index[0, e].item(), self.edge_index[1, e].item()) for e in range(self.num_edges))

    def edges_weighted(self):
        """
        return list of edges where each edge is a tuple ``(source, target, weight)``
        """
        return ((self.edge_index[0, e].item(), self.edge_index[1, e].item(), self.weights[e].cpu().numpy()
                 if self.weights.ndim > 1 else self.weights[e].item()) for e in range(self.num_edges))

    def neighbourhood(self, nodes: torch.Tensor, hops: int = 1):
        """
        find the neighbourhood of a set of source nodes

        note that the neighbourhood includes the source nodes themselves

        Args:
            nodes: indices of source nodes
            hops: number of hops for neighbourhood

        Returns:
            neighbourhood

        """
        explore = torch.ones(self.num_nodes, dtype=torch.bool, device=self.device)
        explore[nodes] = False
        all_nodes = [nodes]
        new_nodes = nodes
        for _ in range(hops):
            new_nodes = torch.cat([self.adj(node) for node in new_nodes])
            new_nodes = torch.unique(new_nodes[explore[new_nodes]])
            explore[new_nodes] = False
            all_nodes.append(new_nodes)
        return torch.cat(all_nodes)

    def subgraph(self, nodes: torch.Tensor, relabel=False, keep_x=True, keep_y=True):
        """
        find induced subgraph for a set of nodes

        Args:
            nodes: node indeces

        Returns:
            subgraph

        """
        index = torch.cat([torch.arange(self.adj_index[node], self.adj_index[node + 1], dtype=torch.long) for node in nodes])
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        node_mask[nodes] = True
        node_ids = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)
        node_ids[nodes] = torch.arange(len(nodes), device=self.device)
        index = index[node_mask[self.edge_index[1][index]]]
        edge_attr = self.edge_attr
        if relabel:
            node_labels = None
        else:
            node_labels = [self.nodes[n] for n in nodes]

        if self.x is not None and keep_x:
            x = self.x[nodes, :]
        else:
            x = None

        if self.y is not None and keep_y:
            y = self.y[nodes]
        else:
            y = None

        return self.__class__(edge_index=node_ids[self.edge_index[:, index]],
                              edge_attr=edge_attr[index] if edge_attr is not None else None,
                              num_nodes=len(nodes),
                              ensure_sorted=False,
                              undir=self.undir,
                              x=x,
                              y=y,
                              nodes=node_labels
                              )

    def connected_component_ids(self):
        """Find the (weakly)-connected components. Component ids are sorted by size, such that id=0 corresponds
         to the largest connected component"""
        edge_index = self.edge_index
        is_undir = self.undir
        last_components = torch.full((self.num_nodes,), self.num_nodes, dtype=torch.long, device=self.device)
        components = torch.arange(self.num_nodes, dtype=torch.long, device=self.device)
        while not torch.equal(last_components, components):
            last_components[:] = components
            components = ts.scatter(last_components[edge_index[0]], edge_index[1], out=components, reduce='min')
            if not is_undir:
                components = ts.scatter(last_components[edge_index[1]], edge_index[0], out=components, reduce='min')
        component_id, inverse, component_size = torch.unique(components, return_counts=True, return_inverse=True)
        new_id = torch.argsort(component_size, descending=True)
        return new_id[inverse]

    def nodes_in_lcc(self):
        """List all nodes in the largest connected component"""
        return torch.nonzero(self.connected_component_ids() == 0).flatten()

    def to_networkx(self):
        """convert graph to NetworkX format"""
        if self.undir:
            nxgraph = nx.Graph()
        else:
            nxgraph = nx.DiGraph()
        nxgraph.add_nodes_from(range(self.num_nodes))
        if self.x is not None:
            for i in range(self.num_nodes):
                nxgraph.nodes[i]['x'] = self.x[i, :]
        if self.y is not None:
            for i in range(self.num_nodes):
                nxgraph.nodes[i]['y'] = self.y[i]
        if self.weighted:
            nxgraph.add_weighted_edges_from(self.edges_weighted())
        else:
            nxgraph.add_edges_from(self.edges())
        return nxgraph

    def to(self, *args, graph_cls=None, device=None):
        """
        Convert to different graph type or move to device

        Args:
            graph_cls: convert to graph class
            device: convert to device

        Can only specify one argument. If positional, type of move is determined automatically.

        """
        if args:
            if not (graph_cls is None and device is None):
                raise ValueError("Both positional and keyword arguments specified.")
            arg, = args
            if isinstance(arg, type) and issubclass(arg, Graph):
                graph_cls = arg
            else:
                device = arg

            if not (graph_cls is None or device is None):
                raise ValueError("Can only specify one of `device` or `graph_cls`")

            if device is not None:
                for key, value in self.__dict__.items():
                    if isinstance(value, torch.Tensor):
                        self.__dict__[key] = value.to(device)
                return self

            if graph_cls is not None:
                return super().to(graph_cls)

    def bfs_order(self, start=0):
        """
        return nodes in breadth-first-search order

        Args:
            start: index of starting node (default: 0)

        Returns:
            tensor of node indeces

        """
        bfs_list = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)
        not_visited = torch.ones(self.num_nodes, dtype=torch.bool, device=self.device)
        bfs_list[0] = start
        not_visited[start] = False
        append_pointer = 1
        i = 0
        while append_pointer < self.num_nodes:
            node = bfs_list[i]
            if node < 0:
                node = torch.nonzero(not_visited)[0]
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

    def partition_graph(self, partition):
        num_clusters = torch.max(partition) + 1
        pe_index = partition[self.edge_index[0]]*num_clusters + partition[self.edge_index[1]]
        partition_edges, weights = torch.unique(pe_index, return_counts=True)
        partition_edges = torch.stack((partition_edges // num_clusters, partition_edges % num_clusters), dim=0)
        return self.__class__(edge_index=partition_edges, edge_attr=weights, num_nodes=num_clusters, undir=self.undir)

    def sample_negative_edges(self, num_samples):
        return tg.utils.negative_sampling(self.edge_index, self.num_nodes, num_samples)

    def sample_positive_edges(self, num_samples):
        index = torch.randint(self.num_edges, (num_samples,), dtype=torch.long)
        return self.edge_index[:, index]
