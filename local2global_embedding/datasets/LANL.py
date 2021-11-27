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
from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
import json

from pandas import DataFrame, read_csv
import numpy as np
import scipy.sparse as ss

fields = ('source_id', 'dest_id', 'total_time', 'src_packets', 'dest_packets', 'src_bytes', 'dst_bytes')


def _build_adj(data: DataFrame, weight=None, weight_transform=None):
    if weight is None:
        weight = np.broadcast_to(np.ones(1), (data.shape[0],))
    else:
        weight = data[weight]

    if weight_transform is not None:
        weight = weight.apply(weight_transform)

    return ss.coo_matrix((weight, (data['source_id'], data['dest_id'])))


class LANLdays:
    def __init__(self, data: LANL, labels=None):
        self._data = data
        if labels is None:
            self.labels = data.timestep_labels
        else:
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return LANLdays(self._data, self.labels[item])
        elif isinstance(item, Iterable):
            return LANLdays(self._data, [self.labels[i] for i in item])
        else:
            data = read_csv(self._data.data_root / 'LANL' / 'netflow' / 'aggregate' / f'netflow_{self.labels[item]}_aggregate_{self._data.protocol}.csv', names=fields)
            return _build_adj(data, self._data.weight, self._data.weight_transform)


class LANL:
    num_source_nodes = 37177
    num_dest_nodes = 931655
    timestep_labels = [f'day-{d:02}' for d in range(2, 91)]

    def __init__(self, data_root, protocol='TCP', weight=None, weight_transform=None):
        self.data_root = Path(data_root)
        self.protocol = protocol
        self.weight = weight
        self.weight_transform = weight_transform
        self.timesteps = LANLdays(self)

    @cached_property
    def source_index(self):
        with open(self.data_root / 'LANL' / 'netflow' / 'aggregate' / 'source_index.json') as f:
            index = json.load(f)
        return index

    @cached_property
    def source_node_labels(self):
        index = self.source_index
        labels = [''] * len(index)
        for key, value in index.items():
            labels[value] = key
        return labels

    @cached_property
    def dest_index(self):
        with open(self.data_root / 'LANL' / 'netflow' / 'aggregate' / 'dest_index.json') as f:
            index = json.load(f)
        return index

    @cached_property
    def dest_node_labels(self):
        index = self.dest_index
        labels = [''] * len(index)
        for key, value in index.items():
            labels[value] = key
        return labels

    @cached_property
    def source_redteam_labels(self):
        data = read_csv(self.data_root / 'LANL' / 'redteam' / 'public_release' / 'redteam_usersip.csv', names=('user', 'comp'))
        return set(data['comp'])

    @cached_property
    def dest_redteam_labels(self):
        data = read_csv(self.data_root / 'LANL' / 'redteam' / 'public_release' / 'redteam_userdip.csv', names=('user', 'comp'))
        return set(data['comp'])
