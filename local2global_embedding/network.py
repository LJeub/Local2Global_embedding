"""Graph data handling"""

import typing as _t
import torch
import torch_geometric as tg
import torch_scatter as ts
from networkx.utils import UnionFind
import networkx as nx


class TGraph:
    """Wrapper class for pytorch-geometric edge_index providing fast adjacency look-up."""

    def __init__(self, edge_index, edge_attr: _t.Optional[torch.Tensor] = None, num_nodes: _t.Optional[int] = None,
                 ensure_sorted: bool = False, undir:_t.Optional[bool] = None):
        """
        Initialise graph

        Args:
            edge_index: edge index such that ``edge_index[0]`` lists the source and ``edge_index[1]`` the target node for each edge
            edge_attr: optionally provide edge weights
            num_nodes: specify number of nodes (default: ``max(edge_index)+1``)
            ensure_sorted: if ``False``, assume that the ``edge_index`` input is already sorted
            undir: boolean indicating if graph is directed. If not provided, the ``edge_index`` is checked to determine this value.
        """
        self.num_nodes = int(torch.max(edge_index)+1) if num_nodes is None else int(num_nodes)  #: number of nodes
        self.num_edges = int(edge_index.shape[1])  #: number of edges
        if ensure_sorted:
            index = torch.argsort(edge_index[0]*self.num_nodes+edge_index[1])
            edge_index = edge_index[:, index]
            if edge_attr is not None:
                edge_attr = edge_attr[index]

        self.edge_index = edge_index  #: edge list (note ``edge_index[0]`` lists the source nodes and ``edge_index[1]`` lists the target nodes)
        self.edge_attr = edge_attr  #: edge weights if weighted, otherwise ``None``
        self.degree = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)  #: tensor of node degrees
        self.degree.index_add_(0, self.edge_index[0],
                               torch.ones(1, dtype=torch.long, device=self.device).expand(self.num_edges))  # use expand to avoid actually allocating large array
        self.adj_index = torch.zeros(self.num_nodes + 1, dtype=torch.long)  #: adjacency index such that edges starting at node ``i`` are given by ``edge_index[:, adj_index[i]:adj_index[i+1]]``
        self.adj_index[1:] = torch.cumsum(self.degree, 0)
        # use expand to avoid actually allocating large array
        self.weights = edge_attr if edge_attr is not None else torch.ones(1, device=self.device).expand(self.num_edges)  #: edge weights
        if self.weighted:
            self.strength = torch.zeros(self.num_nodes, device=self.device)  #: tensor of node strength
            self.strength.index_add_(0, self.edge_index[0], self.weights)
        else:
            self.strength = self.degree

        self.undir = undir if undir is not None else False  #: boolean indicating if graph is undirected
        if undir is None:
            index = torch.argsort(edge_index[1]*self.num_nodes+edge_index[0])
            self.undir = torch.equal(self.edge_index, self.edge_index[:, index].flip((0,)))
            if self.weighted:
                self.undir = self.undir and torch.equal(self.weights, self.weights[index])

    @property
    def device(self):
        """device holding graph data"""
        return self.edge_index.device

    @property
    def weighted(self):
        """boolean indicating if graph is weighted"""
        return self.edge_attr is None

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

    def edges(self):
        """
        return list of edges where each edge is a tuple ``(source, target)``
        """
        return [(e[0].item(), e[1].item()) for e in self.edge_index.T]

    def edges_weighted(self):
        """
        return list of edges where each edge is a tuple ``(source, target, weight)``
        """
        return [(e[0].item(), e[1].item(), w[0].item() if w.numel() > 1 else w.item())
                for e, w in zip(self.edge_index.T, self.weights)]

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

    def subgraph(self, nodes: torch.Tensor):
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
        return TGraph(edge_index=node_ids[self.edge_index[:, index]],
                              edge_attr=edge_attr[index] if edge_attr is not None else None,
                              num_nodes=len(nodes),
                              ensure_sorted=False,
                              undir=self.undir)

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
        if self.weighted:
            nxgraph.add_weighted_edges_from(self.edges_weighted())
        else:
            nxgraph.add_edges_from(self.edges())
        return nxgraph

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


def connected_components(data: tg.data.Data):
    """
    Find the (weakly)-connected components of graph data. Components are sorted by size, such that id=0 corresponds
     to the largest connected component

     Args:
         data: input graph data
     """
    graph = TGraph(data.edge_index)
    return graph.connected_component_ids()


def largest_connected_component(data: tg.data.Data):
    """find largest connected component of data

    Args:
        data: input graph data
    """
    components = connected_components(data)
    nodes = torch.nonzero(components == 0).flatten()
    return induced_subgraph(data, nodes)


def induced_subgraph(data: tg.data.Data, nodes, extend_hops=0):
    """
    find the subgraph induced by the neighbourhood of a set of nodes

    Args:
        data: input graph data
        nodes: set of source nodes
        extend_hops: number of hops for the neighbourhood (default: 0)

    Returns:
        data for induced subgraph
    """
    nodes = torch.as_tensor(nodes, dtype=torch.long)
    if extend_hops > 0:
        nodes, edge_index, node_map, edge_mask = tg.utils.k_hop_subgraph(nodes, num_hops=extend_hops,
                                                                         edge_index=data.edge_index,
                                                                         relabel_nodes=True)
        edge_attr = data.edge_attr[edge_mask, :] if data.edge_attr is not None else None
    else:
        edge_index, edge_attr = tg.utils.subgraph(nodes, data.edge_index, data.edge_attr, relabel_nodes=True)

    subgraph = tg.data.Data(edge_index=edge_index, edge_attr=edge_attr)
    for key, value in data.__dict__.items():
        if not key.startswith('edge'):
            if hasattr(value, 'shape') and value.shape[0] == data.num_nodes:
                setattr(subgraph, key, value[nodes])
            else:
                setattr(subgraph, key, value)
    subgraph.nodes = nodes
    subgraph.num_nodes = len(nodes)
    return subgraph


def conductance(graph: TGraph, source, target=None):
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
