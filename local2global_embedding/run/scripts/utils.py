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
from shutil import copyfile, move
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

from local2global.patch import FilePatch, Patch, MeanAggregatorPatch
from local2global.lazy import LazyCoordinates, LazyMeanAggregatorCoordinates


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
    """class that deletes the named file when it is garbage-collected
    """
    def __init__(self, filename):
        self._finalizer = finalize(self, remove_file, filename)


class ScopedTemporaryFile:
    def __init__(self, *args, **kwargs):
        file = NamedTemporaryFile(*args, delete=False, **kwargs)
        file.close()
        self._finalizer = finalize(self, remove_file, file.name)
        self.name = file.name

    def __fspath__(self):
        return self.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'



def move_to_tmp(patch):
    patch = copy(patch)
    if isinstance(patch, FilePatch):
        old_file = Path(patch.coordinates.filename)
        new_file = NamedTemporaryFile(delete=False, prefix='patch_', suffix='.npy')
        new_file.close()
        new_file = Path(new_file.name)
        copyfile(old_file.resolve(), new_file)
        patch.coordinates.filename = new_file
        patch._finalizer = FileDeleteFinalizer(new_file)  # wrap this in a separate object so it survives copying as
        # each copy retains a reference of this object and thus cleanup only happens if all copies go out of scope
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


def mean_embedding_chunk(out, patches, start, stop):
    dim = out.shape[1]
    out_chunk = np.zeros((stop - start, dim), dtype=np.float32)
    counts = np.zeros((stop-start,), dtype=np.int)
    for patch in patches:
        index = [c for c, i in enumerate(range(start, stop)) if i in patch.index]
        out_chunk[index] += patch.get_coordinates([range(start, stop)[i] for i in index])
        counts[index] += 1
    out[start:stop] = out_chunk / counts[:, None]


def mean_embedding(patches, shape, output_file, use_tmp=True):

    n_nodes, dim = shape
    try:
        if use_tmp:
            work_file = NamedTemporaryFile(delete=False)
            work_file.close()
            work_file = Path(work_file.name)
        else:
            work_file = output_file.with_suffix('.tmp.npy')
        out = open_memmap(work_file, mode='w+', dtype=np.float32, shape=(n_nodes, dim))
        count = np.zeros((n_nodes,), dtype=np.int)
        for patch in tqdm(patches, desc='compute mean embedding output'):
            out[patch.nodes] += patch.coordinates
            count[patch.nodes] += 1
        out /= count[:, None]
        out.flush()
        move(work_file, output_file)
    except Exception as e:
        remove_file(work_file)
        raise e

