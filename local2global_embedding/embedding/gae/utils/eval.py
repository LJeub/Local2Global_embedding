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

import torch
import torch_geometric as tg
from sklearn.metrics import roc_auc_score

from local2global_embedding.embedding.gae.layers.decoders import DistanceDecoder


def reconstruction_auc(coordinates, data, neg_edges=None, dist=False):
    """
    Compute the network reconstruction auc score

    Args:
        coordinates (torch.tensor): embedding to evaluate
        data (tg.utils.data.Data): network data
        neg_edges: edge index for negative edges (optional)
        dist: if ``True``, use distance decoder to evaluate embedding, otherwise use inner-product decoder
              (default: ``False``)

    Returns:
        ROC-AUC for correctly classifying true edges versus non-edges

    By default the function samples the same number of non-edges as there are true edges, such that a score of 0.5
    corresponds to random classification.

    """
    decoder = DistanceDecoder() if dist else tg.nn.InnerProductDecoder()
    if neg_edges is None:
        neg_edges = tg.utils.negative_sampling(data.edge_index, num_nodes=data.num_nodes)
    with torch.no_grad():
        z = torch.cat((decoder(coordinates, data.edge_index, sigmoid=True),
                       decoder(coordinates, neg_edges, sigmoid=True)),
                      dim=0).cpu().numpy()
        y = torch.cat((torch.ones(data.edge_index.shape[1], device='cpu'),
                       torch.zeros(neg_edges.shape[1], device='cpu')),
                      dim=0).numpy()
    return roc_auc_score(y, z)