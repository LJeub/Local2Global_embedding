"""Graph sparsification"""
import math
import warnings

import numpy as np
import scipy as sc
import scipy.sparse
import scipy.sparse.linalg
import torch

from local2global_embedding.network import TGraph, spanning_tree_mask, spanning_tree


def _sample_edges(graph, n_desired_edges, ensure_connected=True):
    if ensure_connected:
        edge_mask = spanning_tree_mask(graph, maximise=True)
        n_desired_edges -= edge_mask.sum()
        unselected_edges = edge_mask.logical_not().nonzero().flatten()
    else:
        edge_mask = torch.zeros(graph.num_edges, dtype=torch.bool, device=graph.device)
        unselected_edges = torch.arange(graph.num_edges, device=graph.device)
    if n_desired_edges > 0:  # check whether we have sufficiently many edges already
        unselected_edge_index = graph.edge_index[:, unselected_edges]
        reversed_index = torch.argsort(unselected_edge_index[1]*graph.num_nodes + unselected_edge_index[0])
        forward_unselected = unselected_edges[unselected_edge_index[0]<unselected_edge_index[1]]
        reverse_unselected = unselected_edges[reversed_index[unselected_edge_index[0]<unselected_edge_index[1]]]
        index = torch.multinomial(graph.weights[forward_unselected].view(1, -1),
                                  n_desired_edges//2, replacement=False).flatten()
        edge_mask[forward_unselected[index]] = True
        edge_mask[reverse_unselected[index]] = True
    return edge_mask


def resistance_sparsify(graph: TGraph, target_mean_degree, ensure_connected=True, epsilon=1e-2):
    """
    Sparsify a graph to have a target mean degree using effective resistance based sampling


    Args:
        graph: input graph
        target_mean_degree: desired mean degree after sparsification
        ensure_connected: if ``True``, first add edges of a maximum spanning tree based on the resistance weights
                          to ensure that the sparsified graph remains connected if the input graph is connected
        epsilon: tolerance for effective resistance computation

    Returns:
        sparsified graph

    This algorithm is based on the method of

        D. A. Spielman and N. Srivastava.
        “Graph sparsification by effective resistances”. SIAM Journal on Computing 40.6 (2011), pp. 1913–1926.

    However, a fixed number of edges are sampled without replacement, and optionally a maximum spanning tree is kept
    to ensure the connectedness of the sparsified graph.

    """
    n_desired_edges = int(target_mean_degree * graph.num_nodes / 2) * 2  # round down to an even number of edges
    if n_desired_edges >= graph.num_edges:
        # graph is already sufficiently sparse
        return graph

    rgraph = resistance_weighted_graph(graph, epsilon=epsilon)

    edge_mask = _sample_edges(rgraph, n_desired_edges, ensure_connected)
    edge_index = graph.edge_index[:, edge_mask]
    edge_attr = None if graph.edge_attr is None else graph.edge_attr[edge_mask]
    return TGraph(edge_index=edge_index, edge_attr=edge_attr, num_nodes=graph.num_nodes, ensure_sorted=False,
                  undir=graph.undir)


def resistance_weighted_graph(graph: TGraph, **args):
    """
    modify the edge weights of a graph by multiplying by their effective resistance

    Args:
        graph: input graph
        epsilon: tolerance for effective resistance computation (default: ``1e-2``)

    Returns:
        copy of input graph with reweighted edges
    """
    resistances = effective_resistances(graph, **args)
    if graph.edge_attr is None:
        edge_attr = resistances
    else:
        edge_attr = graph.edge_attr * resistances
    return TGraph(graph.edge_index, edge_attr, num_nodes=graph.num_nodes, ensure_sorted=False, undir=graph.undir)


def effective_resistances(graph: TGraph, **args):
    """
    compute effective resistances

    Args:
        graph: input graph
        epsilon: tolerance for effective resistance computation (default: ``1e-2``)

    Returns:
        effective resistance for each edge
    """
    Z = _compute_Z(graph, **args)
    Z = torch.from_numpy(Z)
    resistances = torch.pairwise_distance(Z[graph.edge_index[0], :], Z[graph.edge_index[1], :]) ** 2
    return resistances


def _edge_node_incidence_matrix(graph: TGraph):
    indices = np.empty(2 * graph.num_edges, dtype=int)
    values = np.empty(2 * graph.num_edges, dtype=int)
    indptr = 2 * np.arange(graph.num_edges+1, dtype=np.int64)
    indices[::2] = graph.edge_index[0]
    indices[1::2] = graph.edge_index[1]
    values[::2] = 1
    values[1::2] = -1

    return sc.sparse.csr_matrix((values, indices, indptr), shape=(graph.num_edges, graph.num_nodes))


def _edge_weight_matrix(graph: TGraph):
    weight = graph.weights.cpu().numpy()
    W = sc.sparse.dia_matrix((np.sqrt(weight), 0), shape=(len(weight), len(weight)))
    return W


def _compute_Z(graph: TGraph, epsilon=10.0 ** -2.0):
    W = _edge_weight_matrix(graph)
    B = _edge_node_incidence_matrix(graph)
    Y = W.dot(B)
    L = Y.transpose().dot(Y)

    n = graph.num_nodes
    m = graph.num_edges
    k = math.floor(24.0 * math.log(n) / (epsilon ** 2.0))
    delta = epsilon/3.0 * math.sqrt((2.0*(1.0-epsilon)*min(W.diagonal())) / ((1.0+epsilon)*(n**3.0)*max(W.diagonal())))

    LU = sc.sparse.linalg.spilu(L+epsilon*sc.sparse.eye(n))
    P = sc.sparse.linalg.LinearOperator((n, n), matvec=LU.solve)
    Z = np.zeros((n, min(m, k)))

    for i in range(Z.shape[1]):
        if k < m:
            q = ((2 * np.random.randint(0, 2, size=(1, m)) - 1) / math.sqrt(k))
            y = q * B
            y = y.transpose()
        else:
            y = Y.getrow(i).transpose().toarray()

        Z[:, i], flag = sc.sparse.linalg.lgmres(L, y, M=P, tol=delta)

        if flag > 0:
            warnings.warn("BiCGstab not converged after {0} iterations".format(flag))
            print(Z[:, i])

        if flag < 0:
            warnings.warn("BiCGstab error {0}".format(flag))
            print(Z[:, i])

    return Z


def relaxed_spanning_tree(graph: TGraph, maximise=False, gamma=1):
    r"""compute relaxed minimum or maximum spanning tree

    This implements the relaxed minimum spanning tree algorithm of

        M. Beguerisse-Díaz, B. Vangelov, and M. Barahona.
        “Finding role communities in directed networks using Role-Based Similarity, Markov Stability and the Relaxed Minimum Spanning Tree”.
        In: 2013 IEEE Global Conference on Signal and Information Processing (GlobalSIP).
        IEEE, 2013, pp. 937–940. isbn: 978-1-4799-0248-4.

    Args:
        graph: input graph
        maximise: if ``True`` start with maximum spanning tree
        gamma: :math:`\gamma` value for adding edges
    """
    mst = spanning_tree(graph, maximise=maximise)
    rmst_edges = [mst.edge_index]
    rmst_weights = [mst.edge_attr]
    if maximise:
        reduce_fun = torch.minimum
        d = torch.tensor([torch.max(graph.adj_weighted(node)[1]) for node in range(graph.num_nodes)],
                         device=graph.device)
    else:
        reduce_fun = torch.maximum
        d = torch.tensor([torch.min(graph.adj_weighted(node)[1]) for node in range(graph.num_nodes)],
                         device=graph.device)
    target_mask = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=graph.device)
    for i in range(graph.num_nodes):
        neighbours, weights = graph.adj_weighted(i)
        # provide indices into neighbours so we can look up weights easily
        target_mask[neighbours] = torch.arange(neighbours.numel())
        # breadth-first search over mst to find mst path weights (note mst-edges are already added)
        mst_neighbours, mst_weights = mst.adj_weighted(i)
        target_mask[mst_neighbours] = -1
        not_visited = torch.ones(graph.num_nodes, dtype=torch.bool, device=graph.device)
        not_visited[mst_neighbours] = False
        not_visited[i] = False
        while torch.any(target_mask[neighbours] >= 0):
            next_neighbours = []
            next_weights = []
            for node, weight in zip(mst_neighbours, mst_weights):
                n, w = mst.adj_weighted(node)
                new = not_visited[n]
                n = n[new]
                w = w[new]
                not_visited[n] = False
                next_neighbours.append(n)
                next_weights.append(reduce_fun(weight, w))
            mst_neighbours = torch.cat(next_neighbours)
            mst_weights = torch.cat(next_weights)
            index = target_mask[mst_neighbours]
            selected = mst_neighbours[index >= 0]
            target_mask[selected] = -1
            selected_w = mst_weights[index >= 0]
            index = index[index >= 0]
            if maximise:
                add = selected_w-gamma*(d[i]+d[selected]) < weights[index]
            else:
                add = selected_w+gamma*(d[i]+d[selected]) > weights[index]
            rmst_edges.append(torch.stack((torch.full((add.sum().item(),), i, dtype=torch.long), selected[add]), dim=0))
            rmst_weights.append(weights[index[add]])
    edge_index = torch.cat(rmst_edges, dim=1)
    edge_attr = torch.cat(rmst_weights)
    return TGraph(edge_index, edge_attr, graph.num_nodes, ensure_sorted=True, undir=graph.undir)


def edge_sampling_sparsify(graph: TGraph, target_degree, ensure_connected=True):
    n_desired_edges = int(target_degree * graph.num_nodes / 2) * 2  # round down to an even number of edges
    if n_desired_edges >= graph.num_edges:
        # graph is already sufficiently sparse
        return graph

    weights = graph.weights / torch.minimum(graph.strength[graph.edge_index[0]], graph.strength[graph.edge_index[1]])
    cgraph = TGraph(graph.edge_index, edge_attr=weights, adj_index=graph.adj_index, num_nodes=graph.num_nodes,
                   ensure_sorted=False, undir=graph.undir)  # convert weights to conductance value
    edge_mask = _sample_edges(cgraph, n_desired_edges, ensure_connected)
    edge_attr = graph.edge_attr[edge_mask] if graph.edge_attr is not None else None
    return TGraph(edge_index=graph.edge_index[:, edge_mask], edge_attr=edge_attr, num_nodes=graph.num_nodes,
                  ensure_sorted=False, undir=graph.undir)
