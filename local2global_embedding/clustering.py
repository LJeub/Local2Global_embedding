"""Graph clustering algorithms"""

from math import log, sqrt

import community
import torch
import pymetis

from local2global_embedding.network import TGraph


def distributed_clustering(graph: TGraph, beta, rounds=None, patience=3, min_samples=2):
    """
    Distributed clustering algorithm

    Implements algorithm of [#dist]_ with gpu support

    .. [#dist] H. Sun and L. Zanetti. “Distributed Graph Clustering and Sparsification”.
               ACM Transactions on Parallel Computing 6.3 (2019), pp. 1–23.
               doi: 10.1145/3364208. url: https://doi.org/10.1145% 2F3364208.

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


def fennel_clustering(graph: TGraph, num_clusters, load_limit=1.1, alpha=None, gamma=1.5, randomise_order=False,
                      clusters=None, num_iters=1):
    if num_iters is None:
        num_iters = 1

    if alpha is None:
        alpha = graph.num_edges * (num_clusters ** (gamma-1)) / (graph.num_nodes ** gamma)

    partition_sizes = torch.zeros(num_clusters, dtype=torch.long, device=graph.device)
    if clusters is None:
        clusters = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=graph.device)
    else:
        clusters = torch.clone(clusters).to(device=graph.device)
        partition_sizes.index_add_(0, clusters, torch.ones_like(clusters))

    load_limit *= graph.num_nodes/num_clusters

    if randomise_order:
        order = torch.randperm(graph.num_nodes, dtype=torch.long, device=graph.device)
    else:
        order = graph.bfs_order()

    for it in range(num_iters):
        not_converged = 0
        for n in order:
            old_cluster = clusters[n]
            if old_cluster >= 0:
                partition_sizes[old_cluster] -= 1
            deltas = - alpha * gamma * (partition_sizes ** (gamma-1))
            cluster_indices = clusters[graph.adj(n)]
            cluster_indices = cluster_indices[cluster_indices >= 0]
            if cluster_indices.numel() > 0:
                deltas.index_add_(0, cluster_indices, torch.ones(cluster_indices.shape, device=graph.device))
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
    "Implements clustering using the Louvain algorithm for modularity optimisation"
    clusters = community.best_partition(graph.to_networkx().to_undirected(), *args, **kwargs)
    return torch.tensor([clusters[i] for i in range(graph.num_nodes)], dtype=torch.long)


def metis_clustering(graph: TGraph, num_clusters, recursive=False):
    """Implements clustering using metis"""
    adj_list = [graph.adj(i).cpu().numpy() for i in range(graph.num_nodes)]
    n_cuts, memberships = pymetis.part_graph(num_clusters, adjacency=adj_list)
    return torch.as_tensor(memberships, dtype=torch.long, device=graph.device)
