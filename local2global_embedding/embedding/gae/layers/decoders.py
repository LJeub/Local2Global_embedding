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


class DistanceDecoder(torch.nn.Module):
    """
    implements the distance decoder which predicts the probability of an edge as the exponential of the
    negative euclidean distance between nodes
    """
    def __init__(self):
        super(DistanceDecoder, self).__init__()
        self.dist = torch.nn.PairwiseDistance()

    def forward(self, z, edge_index, sigmoid=True):
        """
        compute decoder values

        Args:
            z: input coordinates
            edge_index: edges
            sigmoid: if ``True``, return exponential of negative distance, else return negative distance

        """
        value = -self.dist(z[edge_index[0]], z[edge_index[1]])
        return torch.exp(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        """
        compute value for all node pairs

        Args:
            z: input coordinates
            sigmoid: if ``True``, return exponential of negative distance, else return negative distance

        """
        adj = -torch.cdist(z, z)
        return torch.exp(adj) if sigmoid else adj