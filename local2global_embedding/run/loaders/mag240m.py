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

import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import numba
import numpy as np
from numpy.lib.format import open_memmap

from local2global_embedding import progress as progress
from local2global_embedding.network import NPGraph
from local2global_embedding.run.utils import dataloader, classificationloader

rng = np.random.default_rng()


@numba.njit
def _transform_mag240m(edge_index, undir_index, sort_index, num_nodes):
    with numba.objmode:
        print('pass over edge source, forward')
        progress.reset(edge_index.shape[1])

    for i, e in enumerate(edge_index[0]):
        sort_index[i] = e * num_nodes
        if i % 1000000 == 0 and i > 0:
            with numba.objmode:
                progress.update(1000000)

    with numba.objmode:
        progress.close()
        print('pass over edge source, backward')
        progress.reset(edge_index.shape[1])

    for i, e in enumerate(edge_index[0]):
        sort_index[i+edge_index.shape[1]] = e
        if i % 1000000 == 0 and i > 0:
            with numba.objmode:
                progress.update(1000000)

    with numba.objmode:
        progress.close()
        print('\n pass over edge target, forward\n')
        progress.reset(edge_index.shape[1])

    for i, e in enumerate(edge_index[1]):
        sort_index[i] += e
        if i % 1000000 == 0 and i > 0:
            with numba.objmode:
                progress.update(1000000)
    with numba.objmode:
        progress.close()
        print('\n pass over edge target, forward\n')
        progress.reset(edge_index.shape[1])

    for i, e in enumerate(edge_index[1]):
        sort_index[edge_index.shape[1]+i] += e * num_nodes
        if i % 1000000 == 0 and i > 0:
            with numba.objmode:
                progress.update(1000000)

    with numba.objmode:
        progress.close()
        print('\n sorting edge_index\n')
        sort_index.sort()
        print('storing undirected edges')
        progress.reset(sort_index.size - 1)

    num_edges = 1
    undir_index[:, 0] = divmod(sort_index[0], num_nodes)
    for it, index in enumerate(sort_index[1:]):
        if sort_index[it] != index:
            # bidirectional edges in the original data will be duplicated and need to be removed
            undir_index[:, num_edges] = divmod(sort_index[it], num_nodes)
            num_edges += 1
        if it % 1000000 == 0 and it > 0:
            with numba.objmode:
                progress.update(1000000)

    with numba.objmode:
        progress.close()
    return num_edges


@dataloader('MAG240M')
def _load_mag240(root='.', mmap_features='r', mmap_edges='r', load_features=True, **kwargs):
    root = Path(root)
    data_folder = root / 'mag240m_citations_undir'
    if not data_folder.is_dir() or not (data_folder / 'processed').is_file():
        data_folder.mkdir(parents=True, exist_ok=True)
        from ogb.lsc import MAG240MDataset
        base_data = MAG240MDataset(root=root)
        num_nodes = base_data.num_papers
        edge_index = np.load(root / 'mag240m_kddcup2021' / 'processed' / 'paper___cites___paper' / 'edge_index.npy',
                             mmap_mode='r')
        undir_index_file = data_folder / 'edge_index.npy'
        if undir_index_file.is_file():
            undir_index = open_memmap(undir_index_file, mode='r+')
        else:
            undir_index = open_memmap(undir_index_file, mode='w+',
                                      shape=(edge_index.shape[0], 2*edge_index.shape[1]), dtype=np.int64,
                                      fortran_order=True)
        if np.array_equal(undir_index[:, -1], [0, 0]):
            sort_index_file = NamedTemporaryFile(delete=False, suffix='.npy')
            sort_index_file.close()
            sort_index = open_memmap(sort_index_file.name, dtype='i8',
                                     shape=(undir_index.shape[1],),
                                     mode='w+')
            num_edges = _transform_mag240m(edge_index, undir_index, sort_index, num_nodes)
            undir_index = undir_index[:, :num_edges]
            f = NamedTemporaryFile(delete=False)
            np.save(f, undir_index)
            f.close()
            shutil.copy(f.name, undir_index_file)
            del sort_index
            Path(sort_index_file.name).unlink()
            Path(f.name).unlink()

        with open(data_folder / 'info.json', 'w') as f:
            json.dump({'num_nodes': num_nodes, 'undir': True}, f)

        feat_file = data_folder / 'node_feat.npy'
        if not feat_file.is_file():
            print('link node features')
            (root / 'mag240m_kddcup2021' / 'processed' / 'paper' / 'node_feat.npy').link_to(feat_file)

        label_file = data_folder / 'node_label.npy'
        if not label_file.is_file():
            print('link node labels')
            (root / 'mag240m_kddcup2021' / 'processed' / 'paper' / 'node_label.npy').link_to(label_file)

        (data_folder / 'processed').touch()

    index_file = data_folder / 'adj_index.npy'

    if load_features:
        data = NPGraph.load(data_folder, mmap_edges=mmap_edges, mmap_features=mmap_features)
    else:
        edge_index = np.load(data_folder / 'edge_index.npy', mmap_mode=mmap_edges)
        if index_file.is_file():
            adj_index = np.load(index_file)
        else:
            adj_index = None

        with open(data_folder / 'info.json') as f:
            kwargs = json.load(f)
        data = NPGraph(edge_index=edge_index, adj_index=adj_index, ensure_sorted=False, **kwargs)

    if not index_file.is_file():
        np.save(index_file, data.adj_index)

    return data


@classificationloader
def _load_mag240m_classification(root='/tmp', num_val=10000, **kwargs):
    from ogb.lsc import MAG240MDataset
    base_data = MAG240MDataset(root=root)
    y = base_data.all_paper_label
    train = base_data.get_idx_split('train')
    val_test = base_data.get_idx_split('valid')
    val = rng.choice(val_test, size=num_val, replace=False, shuffle=False)
    test = np.delete(val_test, val)
    return y, {'train': train, 'val': val, 'test': test}
