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

import torch_geometric as tg

from ..utils.mixins import EmbeddingMixin
from ..layers.decoders import DistanceDecoder
from ..layers.VGAEconv import VGAEconv


class VGAE(tg.nn.VGAE, EmbeddingMixin):
    def __init__(self, dim, hidden_dim, num_features, dist=False):
        """
        initialise a Variational Graph Auto-Encoder model

        Args:
            dim: output dimension
            hidden_dim: inner hidden dimension
            num_features: number of input features
            dist: if ``True`` use distance decoder, otherwise use inner product decoder (default: ``False``)

        Returns:
            initialised :class:`tg.nn.VGAE` model
        """
        if dist:
            super().__init__(encoder=VGAEconv(dim, num_node_features=num_features, hidden_dim=hidden_dim),
                             decoder=DistanceDecoder())
        else:
            super().__init__(encoder=VGAEconv(dim, num_node_features=num_features, hidden_dim=hidden_dim))