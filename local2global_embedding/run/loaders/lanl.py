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
from pathlib import Path

from pandas import DataFrame, read_csv
import numpy as np
import scipy.sparse as ss

from local2global_embedding.run.utils import dataloader

fields= ('source_id', 'dest_id', 'total_time', 'src_packets', 'dest_packets', 'src_bytes', 'dst_bytes')


def _build_adj(data: DataFrame, weight=None, weight_transform=None):
    if weight is None:
        weight = np.broadcast_to(np.ones(1), (data.shape[0],))
    else:
        weight = data[weight]

    if weight_transform is not None:
        weight = weight.apply(weight_transform)

    return ss.coo_matrix((weight, (data['source_id'], data['dest_id'])))


@dataloader('lanl')
def _load_data(root, days, protocol='TCP', weight=None, weight_transform=None):
    root = Path(root)
    data = (read_csv(root / f'netflow_day-{day:02}_aggregate_{protocol}.csv', names=fields) for day in days)
    return (_build_adj(d, weight, weight_transform) for d in data)