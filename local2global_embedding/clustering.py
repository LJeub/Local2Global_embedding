"""Graph clustering algorithms"""

from math import log
import os

import community
import torch
import pymetis
import numpy as np
from tqdm.auto import tqdm


from local2global_embedding.network import TGraph


class NodeStream:
    def __init__(self, edge_index, num_nodes):
        """
        Initialize node-stream data

        Args:
            edge_index: Edge index to stream (can be a file path to a .npy file which is then memory mapped)
            num_nodes: number of nodes

        Note that edge index needs to be sorted!
        """
        if isinstance(edge_index, str) or isinstance(edge_index, os.PathLike):
            self._data = np.load(edge_index, mmap_mode='r')
        else:
            self._data = edge_index
        self.num_nodes = num_nodes

    def __iter__(self):
        current_node = 0
        neighbours = []
        for i in range(self._data.shape[1]):
            edge = self._data[:, i]
            if current_node == edge[0]:
                neighbours.append(edge[1])
            else:
                yield current_node, neighbours  # all neighbours accumulated
                for missing_node in range(current_node+1, edge[0]):
                    yield missing_node, []  # output nodes with degree 0
                current_node = edge[0]
                neighbours = [edge[1]]
        yield current_node, neighbours  # output last node with edges
        for missing_node in range(current_node+1, self.num_nodes):
            yield missing_node, []  # output any remaining nodes of degree 0

    @property
    def num_edges(self):
        return self._data.shape[1]

    @property
    def device(self):
        if hasattr(self._data, 'device'):
            return self._data.device
        else:
            return 'cpu'



def distributed_clustering(graph: TGraph, beta, rounds=None, patience=3, min_samples=2):
    r"""
    Distributed clustering algorithm

    Implements algorithm of [#dist]_ with gpu support

    Args:
        graph: input graph
        beta: :math:`\beta` value of the algorithm (controls the number of seeds)
        rounds: number of iteration rounds (default: ``3*int(log(graph.num_nodes))``)
        patience: number of rounds without label changes before early stopping (default: ``3``)
        min_samples: minimum number of seed nodes (default: ``2``)

    .. Rubric:: Reference

    .. [#dist] H. Sun and L. Zanetti. “Distributed Graph Clustering and Sparsification”.
               ACM Transactions on Parallel Computing 6.3 (2019), pp. 1–23.
               doi: `10.1145/3364208 <https://doi.org/10.1145/3364208>`_.

    """
    if rounds is None:
        rounds = 3*int(log(graph.num_nodes))
    strength = graph.strength

    # sample seed nodes
    index = torch.rand((graph.num_nodes,)) < 1/beta * log(1 / beta) * graph.strength / graph.strength.sum()
    while index.sum() < min_samples:
        index = torch.rand((graph.num_nodes,)) < 1/beta * log(1 / beta) * graph.strength / graph.strength.sum()
    seeds = torch.nonzero(index).flatten()
    n_samples = seeds.numel()

    states = torch.zeros((graph.num_nodes, n_samples), dtype=torch.double, device=graph.device)
    states[index, torch.arange(n_samples, device=graph.device)] = 1/torch.sqrt(strength[index]).to(dtype=torch.double)
    clusters = torch.argmax(states, dim=1)
    weights = graph.weights / torch.sqrt(strength[graph.edge_index[0]]*strength[graph.edge_index[1]])
    weights = weights.to(dtype=torch.double)
    r = 0
    num_same = 0
    while r < rounds and num_same < patience:  # keep iterating until clustering does not change for 'patience' rounds
        r += 1
        states *= 0.5
        states.index_add_(0, graph.edge_index[0], 0.5*states[graph.edge_index[1]]*weights.view(-1, 1))
        # states = ts.scatter(out=states, dim=0, index=graph.edge_index[0],
        #                     src=0.5*states[graph.edge_index[1]]*weights.view(-1, 1))
        old_clusters = clusters
        clusters = torch.argmax(states, dim=1)
        if torch.equal(old_clusters, clusters):
            num_same += 1
        else:
            num_same = 0
    clusters[states[range(graph.num_nodes), clusters] == 0] = -1
    uc, clusters = torch.unique(clusters, return_inverse=True)
    if uc[0] == -1:
        clusters -= 1
    return clusters


def fennel_clustering(nodes: NodeStream, num_clusters, load_limit=1.1, alpha=None, gamma=1.5, randomise_order=False,
                      clusters=None, num_iters=1):
    r"""
    FENNEL single-pass graph clustering algorithm

    Implements the graph clustering algorithm of [#fennel]_.

    Args:
        graph: input graph
        num_clusters: target number of clusters
        load_limit: maximum cluster size is ``load_limit * graph.num_nodes / num_clusters`` (default: ``1.1``)
        alpha: :math:`\alpha` value for the algorithm (default as suggested in [#fennel]_)
        gamma: :math:`\gamma` value for the algorithm (default: 1.5)
        randomise_order: if ``True``, randomise order, else use breadth-first-search order.
        clusters: input clustering to refine (optional)
        num_iters: number of cluster assignment iterations (default: ``1``)

    Returns:
        cluster index tensor

    References:
        .. [#fennel] C. Tsourakakis et al. “FENNEL: Streaming Graph Partitioning for Massive Scale Graphs”.
                     In: Proceedings of the 7th ACM international conference on Web search and data mining.
                     WSDM'14 (2014) doi: `10.1145/2556195.2556213 <https://doi.org/10.1145/2556195.2556213>`_.

    """
    device = nodes.device

    if num_iters is None:
        num_iters = 1

    if alpha is None:
        alpha = nodes.num_edges * (num_clusters ** (gamma-1)) / (nodes.num_nodes ** gamma)

    partition_sizes = torch.zeros(num_clusters, dtype=torch.long, device=device)
    if clusters is None:
        clusters = torch.full((nodes.num_nodes,), -1, dtype=torch.long, device=device)
    else:
        clusters = torch.clone(clusters).to(device=device)
        partition_sizes.index_add_(0, clusters, torch.ones_like(clusters))

    load_limit *= nodes.num_nodes/num_clusters

    for it in range(num_iters):
        not_converged = 0
        for n, neighbours in tqdm(nodes):
            old_cluster = clusters[n]
            if old_cluster >= 0:
                partition_sizes[old_cluster] -= 1
            deltas = - alpha * gamma * (partition_sizes ** (gamma-1))
            cluster_indices = clusters[neighbours]
            cluster_indices = cluster_indices[cluster_indices >= 0]
            if cluster_indices.numel() > 0:
                deltas.index_add_(0, cluster_indices, torch.ones(cluster_indices.shape, device=device))
                deltas[partition_sizes >= load_limit] = -float('inf')
            # ind = torch.multinomial((deltas == deltas.max()).float(), 1)
            ind = torch.argmax(deltas)
            if ind != old_cluster:
                not_converged += 1
            clusters[n] = ind
            partition_sizes[ind] += 1

        print(f'iteration: {it}, not converged: {not_converged}')

        if not_converged == 0:
            print(f'converged after {it} iterations.')
            break
    return clusters


def louvain_clustering(graph: TGraph, *args, **kwargs):
    r"""
    Implements clustering using the Louvain [#l]_ algorithm for modularity optimisation

    Args:
        graph: input graph

    Returns:
        partition tensor

    This is a minimal wrapper around :py:func:`community.best_partition` from the
    `python-louvain <https://github.com/taynaud/python-louvain>`_ package. Any other
    arguments provided are passed through.

    References:
        .. [#l] V. D. Blondel et al.
                “Fast unfolding of communities in large networks”.
                Journal of Statistical Mechanics: Theory and Experiment 2008.10 (2008), P10008.
                DOI: `10.1088/1742-5468/2008/10/P10008 <https://doi.org/10.1088/1742-5468/2008/10/P10008>`_

    """
    clusters = community.best_partition(graph.to_networkx().to_undirected(), *args, **kwargs)
    return torch.tensor([clusters[i] for i in range(graph.num_nodes)], dtype=torch.long)


def metis_clustering(graph: TGraph, num_clusters):
    """
    Implements clustering using metis

    Args:
        graph: input graph
        num_clusters: number of cluster

    Returns:
        partition tensor

    This uses the `pymetis <https://github.com/inducer/pymetis>`_ package

    References:
        .. [#metis] “A Fast and Highly Quality Multilevel Scheme for Partitioning Irregular Graphs”.
                    George Karypis and Vipin Kumar.
                    SIAM Journal on Scientific Computing, Vol. 20, No. 1, pp. 359—392, 1999.
    """
    adj_list = [graph.adj(i).cpu().numpy() for i in range(graph.num_nodes)]
    n_cuts, memberships = pymetis.part_graph(num_clusters, adjacency=adj_list)
    return torch.as_tensor(memberships, dtype=torch.long, device=graph.device)
