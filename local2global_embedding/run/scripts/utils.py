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
from tempfile import gettempdir
from filelock import FileLock, SoftFileLock

import numpy as np
from pathlib import Path

from numpy.lib.format import open_memmap
from tqdm.auto import tqdm
from dask.distributed import worker_client, as_completed

from local2global.utils import FilePatch, Patch, MeanAggregatorPatch
from local2global.utils.lazy import LazyCoordinates, LazyMeanAggregatorCoordinates


def load_patch(patch_folder, i, basename, dim, criterion):
    nodes = np.load(patch_folder / f'patch{i}_index.npy')
    coords = np.load(patch_folder / f'{basename}_patch{i}_d{dim}_best_{criterion}_coords.npy')
    return Patch(nodes, LazyCoordinates(coords))


def load_file_patch(patch_folder, i, basename, dim, criterion):
    nodes = np.load(patch_folder / f'patch{i}_index.npy')
    return FilePatch(nodes, patch_folder / f'{basename}_patch{i}_d{dim}_best_{criterion}_coords.npy')


def load_patches(patch_graph, patch_folder, basename, dim, criterion, lazy=True):
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


def move_to_tmp(patch):
    tmpdir = gettempdir()
    patch = copy(patch)
    if isinstance(patch, FilePatch):
        old_file = Path(patch.coordinates.filename)
        new_file = Path(tmpdir) / old_file
        with FileLock(new_file.with_suffix('.lock')):
            if not new_file.is_file():
                new_file.parent.mkdir(parents=True, exist_ok=True)
                copyfile(old_file.resolve(), new_file)
        patch.coordinates.filename = new_file
    elif isinstance(patch, MeanAggregatorPatch):
        patch.coordinates.patches = [move_to_tmp(p) for p in patch.coordinates.patches]
    return patch


def restore_from_tmp(patch):
    tmpdir = gettempdir()
    if isinstance(patch, FilePatch):
        patch.coordinates.filename = Path(patch.coordinates.filename).relative_to(tmpdir)
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


def add_count(counts, patch):
    counts[patch.nodes] += 1
    return counts


def compute(task):
    return task.compute()


def mean_embedding_chunk(output_file, coords: LazyMeanAggregatorCoordinates, start, stop, use_tmp=True):
    if use_tmp:
        coords = copy(coords)
        coords.patches = [move_to_tmp(p) for p in coords.patches]
    out = np.load(output_file, mmap_mode='r+')
    out[start:stop] = coords[start:stop]
    out.flush()


def mean_embedding(coords: LazyMeanAggregatorCoordinates, output_file, use_tmp=True):
    chunk_size = 10000
    n_nodes = coords.shape[0]
    work_file = output_file.with_suffix('.tmp.npy')
    out = open_memmap(work_file, mode='w+', dtype=np.float32, shape=coords.shape)
    out.flush()
    with worker_client() as client:
        coords = client.scatter(coords)
        tasks = []
        for start in range(0, n_nodes,  chunk_size):
            stop = min(start + chunk_size, n_nodes)
            tasks.append(client.submit(mean_embedding_chunk, work_file, coords, start, stop, use_tmp))

        total = len(tasks)
        tasks = as_completed(tasks)
        for _ in tqdm(tasks, 'distributed mean embedding', total=total):
            pass
    work_file.replace(output_file)
