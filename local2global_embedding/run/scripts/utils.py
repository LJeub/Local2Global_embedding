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
from filelock import FileLock

import numpy as np
from pathlib import Path

from local2global.utils import FilePatch, Patch, MeanAggregatorPatch
from local2global.utils.lazy import LazyCoordinates


def load_patches(patch_graph, patch_folder, basename, dim, criterion, lazy=True):
    patches = []
    patch_folder = Path(patch_folder)
    if patch_folder.is_absolute():
        patch_folder = patch_folder.relative_to(Path.cwd())  # make relative path such that use_tmp works correctly
    for i in range(patch_graph.num_nodes):
        nodes = np.load(patch_folder / f'patch{i}_index.npy')
        if lazy:
            patches.append(FilePatch(nodes, patch_folder / f'{basename}_patch{i}_d{dim}_best_{criterion}_coords.npy'))
        else:
            coords = np.load(patch_folder / f'{basename}_patch{i}_d{dim}_best_{criterion}_coords.npy')
            patches.append(Patch(nodes, LazyCoordinates(coords)))
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
