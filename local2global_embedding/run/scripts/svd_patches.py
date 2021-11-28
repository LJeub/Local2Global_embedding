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

import numpy as np

from local2global.patch import FilePatch
from local2global.lazy import LazyFileCoordinates

from local2global_embedding.run.utils import ScriptParser
from local2global_embedding.embedding.svd import bipartite_svd_patches


def svd_patches(data, index, output_folder, dim):
    label = data.timestep_labels[index]
    output_folder = Path(output_folder)

    index_files = [output_folder / f'{label}_{t}_index.npy' for t in ('source', 'dest')]
    coords_files = [output_folder / f'{label}_{t}_svd_{dim}_coords.npy' for t in ('source', 'dest')]
    if not all(f.is_file() for f in coords_files):
        adj = data.timesteps[index]
        patches = bipartite_svd_patches(adj, dim)
        for p, index_file, coords_file in zip(patches, index_files, coords_files):
            if not index_file.is_file():
                np.save(index_file, p.nodes)
            np.save(coords_file, p.coordinates)
            p.coordinates = LazyFileCoordinates(coords_file)
    else:
        patches = [FilePatch(np.load(index_file), coords_file) for index_file, coords_file in zip(index_files, coords_files)]
    return patches
