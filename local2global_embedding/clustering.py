"""Graph clustering algorithms"""

from math import log
import os

import community
import torch
import pymetis
import numpy as np
from tqdm.auto import tqdm
import numba

from local2global_embedding.network import TGraph, NPGraph
from local2global_embedding import progress


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


def fennel_clustering(graph, num_clusters, load_limit=1.1, alpha=None, gamma=1.5, num_iters=1, clusters=None):
    graph = graph.to(NPGraph)

    if clusters is None:
        clusters = _fennel_clustering(graph.edge_index, graph.adj_index, graph.num_nodes, num_clusters, load_limit, alpha, gamma, num_iters)
    else:
        clusters = _fennel_clustering(graph.edge_index, graph.adj_index, graph.num_nodes, num_clusters, load_limit, alpha, gamma, num_iters,
                                      clusters)
    return torch.as_tensor(clusters)


@numba.njit
def _fennel_clustering(edge_index, adj_index, num_nodes, num_clusters, load_limit=1.1, alpha=None, gamma=1.5, num_iters=1,
                       clusters=np.empty(0, dtype=np.int64)):
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
    if num_iters is None:
        num_iters = 1

    num_edges = edge_index.shape[1]
    total = num_edges * num_iters

    if alpha is None:
        alpha = num_edges * (num_clusters ** (gamma-1)) / (num_nodes ** gamma)

    partition_sizes = np.zeros(num_clusters, dtype=np.int64)
    if clusters.size == 0:
        clusters = np.full((num_nodes,), -1, dtype=np.int64)
    else:
        clusters = np.copy(clusters)
        for index in clusters:
            partition_sizes[index] += 1

    load_limit *= num_nodes/num_clusters

    deltas = - alpha * gamma * (partition_sizes ** (gamma - 1))

    with numba.objmode:
        progress.reset(num_nodes)

    for it in range(num_iters):
        not_converged = 0

        progress_it = 0
        for i in range(num_nodes):
            cluster_indices = np.empty((adj_index[i+1]-adj_index[i],), dtype=np.int64)
            for ni, index in enumerate(range(adj_index[i], adj_index[i+1])):
                cluster_indices[ni] = clusters[edge_index[1, index]]
            old_cluster = clusters[i]
            if old_cluster >= 0:
                partition_sizes[old_cluster] -= 1
            cluster_indices = cluster_indices[cluster_indices >= 0]

            if cluster_indices.size > 0:
                c_size = np.zeros(num_clusters, dtype=np.int64)
                for index in cluster_indices:
                    c_size[index] += 1
                ind = np.argmax(deltas + c_size)
            else:
                ind = np.argmax(deltas)
            clusters[i] = ind
            partition_sizes[ind] += 1
            if partition_sizes[ind] == load_limit:
                deltas[ind] = - np.inf
            else:
                deltas[ind] = - alpha * gamma * (partition_sizes[ind] ** (gamma - 1))
            not_converged += ind != old_cluster

            if i % 10000 == 0 and i > 0:
                progress_it = i
                with numba.objmode:
                    progress.update(10000)
        with numba.objmode:
            progress.update(num_nodes - progress_it)

        print('iteration: ' + str(it) + ', not converged: ' + str(not_converged))

        if not_converged == 0:
            print(f'converged after ' + str(it) + ' iterations.')
            break
    with numba.objmode:
        progress.close()

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
