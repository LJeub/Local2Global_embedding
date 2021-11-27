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
import numpy as np
from scipy.special import expit
import torch
from sklearn.metrics import roc_auc_score
from local2global_embedding.network import Graph


def reconstruction_auc(coordinates, graph: Graph, dist=False, max_samples=int(1e6)):
    """
    Compute the network reconstruction auc score

    Args:
        coordinates (torch.tensor): embedding to evaluate
        graph: network data
        dist: if ``True``, use distance decoder to evaluate embedding, otherwise use inner-product decoder
              (default: ``False``)
        max_samples: maximum number of edges to use for evaluation. If graph has less than ``max_samples``
                     edges, all edges are used as positive examples,
                     otherwise, max_samples edges are sampled with replacement. In both cases, the number of negative
                     samples is the same as positive samples.

    Returns:
        ROC-AUC for correctly classifying true edges versus non-edges

    By default the function samples the same number of non-edges as there are true edges, such that a score of 0.5
    corresponds to random classification.

    """
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.cpu().numpy()

    if graph.num_edges > max_samples:
        pos_edges = graph.sample_positive_edges(max_samples)
        num_samples = max_samples
    else:
        pos_edges = graph.edge_index
        num_samples = graph.num_edges
    neg_edges = graph.sample_negative_edges(num_samples)

    if isinstance(pos_edges, torch.Tensor):
        pos_edges = pos_edges.cpu().numpy()

    if isinstance(neg_edges, torch.Tensor):
        neg_edges = neg_edges.cpu().numpy()

    pos_edges = np.asanyarray(pos_edges)
    neg_edges = np.asanyarray(neg_edges)
    coordinates = np.asanyarray(coordinates)
    if dist:
        z = np.concatenate((np.linalg.norm(coordinates[pos_edges[0]]-coordinates[pos_edges[1]], axis=1),
                            np.linalg.norm(coordinates[neg_edges[0]]-coordinates[neg_edges[1]], axis=1)))

        z = np.exp(-z)
    else:
        z = np.concatenate((np.sum(coordinates[pos_edges[0]] * coordinates[pos_edges[1]], axis=1),
                            np.sum(coordinates[neg_edges[0]] * coordinates[neg_edges[1]], axis=1)))
        z = expit(z)
        y = np.concatenate((np.ones(pos_edges.shape[1]), np.zeros(neg_edges.shape[1])))
    return roc_auc_score(y, z)
