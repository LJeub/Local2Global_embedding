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
from copy import copy
from shutil import copyfile
from tempfile import gettempdir, NamedTemporaryFile
from operator import add

import dask.bag
from filelock import FileLock, SoftFileLock
import os
from threading import Lock
from weakref import finalize

import numpy as np
from pathlib import Path

from numpy.lib.format import open_memmap
from tqdm.auto import tqdm
from dask.distributed import worker_client, as_completed, secede, rejoin
from dask import delayed

from local2global.utils import FilePatch, Patch, MeanAggregatorPatch
from local2global.utils.lazy import LazyCoordinates, LazyMeanAggregatorCoordinates


@delayed
def load_patch(patch_folder, i, basename, dim, criterion):
    nodes = np.load(patch_folder / f'patch{i}_index.npy')
    coords = np.load(patch_folder / f'{basename}_patch{i}_d{dim}_best_{criterion}_coords.npy')
    return Patch(nodes, LazyCoordinates(coords))


@delayed
def load_file_patch(patch_folder, i, basename, dim, criterion):
    nodes = np.load(patch_folder / f'patch{i}_index.npy')
    patch = FilePatch(nodes, patch_folder / f'{basename}_patch{i}_d{dim}_best_{criterion}_coords.npy')
    return patch


def load_patches(patch_graph, patch_folder, basename, dim, criterion, lazy=True, use_tmp=False):
    patches = []
    patch_folder = Path(patch_folder)
    if patch_folder.is_absolute():
        patch_folder = patch_folder.relative_to(Path.cwd())  # make relative path such that use_tmp works correctly
    for i in tqdm(range(patch_graph.num_nodes), desc='load patches'):
        if lazy:
            patches.append(load_file_patch(patch_folder, i, basename, dim, criterion))
        else:
            patches.append(load_patch(patch_folder, i, basename, dim, criterion))
    return patches


def remove_file(name):
    try:
        os.unlink(name)
    except FileNotFoundError:
        pass


class FileDeleteFinalizer:
    def __init__(self, filename):
        self._finalizer = finalize(self, remove_file, filename)


def move_to_tmp(patch):
    patch = copy(patch)
    if isinstance(patch, FilePatch):
        old_file = Path(patch.coordinates.filename)
        new_file = NamedTemporaryFile(delete=False, prefix='patch_', suffix='.npy')
        new_file.close()
        new_file = Path(new_file.name)
        copyfile(old_file.resolve(), new_file)
        patch.coordinates.filename = new_file
        patch._finalizer = FileDeleteFinalizer(new_file)  # wrap this in a separate object so it survives copying
        patch.old_file = old_file
    elif isinstance(patch, MeanAggregatorPatch):
        patch.coordinates.patches = [move_to_tmp(p) for p in patch.coordinates.patches]
    return patch


def restore_from_tmp(patch):
    if isinstance(patch, FilePatch):
        tmp_file = Path(patch.coordinates.filename)
        patch.coordinates.filename = patch.old_file
        del patch.old_file
        tmp_file.unlink()
    elif isinstance(patch, MeanAggregatorPatch):
        patch.coordinates.patches = [restore_from_tmp(p) for p in patch.coordinates.patches]
    return patch


def accumulate(patches, file):
    file = Path(file)
    out = np.load(file, mmap_mode='r+')
    for patch in patches:
        coords = np.asarray(patch.coordinates)
        with SoftFileLock(file.with_suffix('.lock')):
            out[patch.nodes] += coords
            out.flush()


def max_ind(patch):
    return patch.nodes.max()


def count_chunk(patch, start, stop):
    counts = np.array([n in patch.index for n in range(start, stop)])
    return counts



def compute(task):
    return task.compute()


def get_coordinate_chunk(patch: Patch, start, stop):
    out = np.zeros((stop-start, patch.shape[1]), dtype=np.float32)
    index = [c for c, i in enumerate(range(start, stop)) if i in patch.index]
    out[index] = patch.get_coordinates([range(start, stop)[i] for i in index])
    return out


@delayed
def get_n_nodes(patches):
    return max(p.nodes.max() for p in patches) + 1


def get_dim(patch):
    return patch.shape[1]


def apply_partition(func):
    def apply(partition, *args, **kwargs):
        return [func(p, *args, **kwargs) for p in partition]
    return apply



def mean_embedding_chunk(file, patch_bag, start, stop):
    out = np.load(file, mmap_mode='r+')
    dim = out.shape[1]
    out_chunk = patch_bag.map(get_coordinate_chunk, start, stop).fold(add, add,
                                                                      initial=np.zeros((stop - start, dim),
                                                                                       dtype=np.float32)).compute()
    counts = patch_bag.map(count_chunk, start, stop).fold(add, add,
                                                          initial=np.zeros((stop - start,), dtype=np.int)).compute()
    out[start:stop] = out_chunk / counts[:, None]


def mean_embedding(patch_bag: dask.bag, shape, output_file, use_tmp=True):
    chunk_size = 100000

    if use_tmp:
        patch_bag = patch_bag.map_partitions(apply_partition(move_to_tmp))
    n_nodes, dim = shape
    work_file = output_file.with_suffix('.tmp.npy')
    out = open_memmap(work_file, mode='w+', dtype=np.float32, shape=(n_nodes, dim))
    chunks = []
    with worker_client() as client:
        for start in range(0, n_nodes,  chunk_size):
            stop = min(start + chunk_size, n_nodes)
            chunks.append(client.submit(mean_embedding_chunk, work_file, patch_bag, start, stop))
        chunks = as_completed(chunks)
        for c in tqdm(chunks, total=chunks.count(), desc='distributed mean embedding'):
            del c
    work_file.replace(output_file)
