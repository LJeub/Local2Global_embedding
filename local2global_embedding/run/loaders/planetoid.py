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

from local2global_embedding.network import TGraph
from local2global_embedding.run.utils import dataloader, classificationloader


def _load_data(name):
    def _load(root='/tmp'):
        return TGraph.from_tg(tg.datasets.Planetoid(name=name, root=f'{root}/{name}')[0])
    return _load


def _load_class(name):
    def _load(root='/tmp'):
        data = tg.datasets.Planetoid(name=name, root=f'{root}/{name}', split='public')[0]
        y = data.y
        split = {'test': data.test_mask.nonzero().flatten(),
                 'train': data.train_mask.nonzero().flatten(),
                 'val': data.val_mask.nonzero().flatten()}
        return y, split
    return _load


for name in ('Cora', 'CiteSeer', 'PubMed'):
    dataloader(name)(_load_data(name))
    classificationloader(name)(_load_class(name))
