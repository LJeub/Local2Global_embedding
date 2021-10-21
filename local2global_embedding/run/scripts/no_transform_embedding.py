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
from tempfile import NamedTemporaryFile
from shutil import copyfile, move

import numpy as np
from numpy.lib.format import open_memmap

from local2global.utils.lazy import LazyMeanAggregatorCoordinates
from .utils import load_patches


def no_transform_embedding(patch_graph, patch_folder, basename, dim, criterion, mmap=True, use_tmp=True):
    patch_folder = Path(patch_folder)
    patches = load_patches(patch_graph, patch_folder, basename, dim, criterion, lazy=mmap)
    output_file = patch_folder / f'{basename}_d{dim}_nt_{criterion}_coords.npy'
    coords = LazyMeanAggregatorCoordinates(patches)
    if mmap:
        if use_tmp:
            with NamedTemporaryFile(delete=False) as f:
                out = np.memmap(f, shape=coords.shape, dtype=np.float32)
                coords.as_array(out)
            move(f.name, output_file, copy_function=copyfile)
        else:
            out = open_memmap(output_file, mode='w+', dtype=np.float32, shape=coords.shape)
            coords.as_array(out)
            out.flush()
    return output_file