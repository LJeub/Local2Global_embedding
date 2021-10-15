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

from local2global.utils import FilePatch
from local2global import Patch


def load_patches(patch_graph, patch_folder, basename, dim, criterion, lazy=True):
    patches = []
    for i in range(patch_graph.num_nodes):
        nodes = np.load(patch_folder / f'patch{i}_index.npy')
        if lazy:
            patches.append(FilePatch(nodes, patch_folder / f'{basename}_patch{i}_d{dim}_best_{criterion}_coords.npy'))
        else:
            coords = np.load(patch_folder / f'{basename}_patch{i}_d{dim}_best_{criterion}_coords.npy')
            patches.append(Patch(nodes, coords))
    return patches
