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

import torch

from local2global import WeightedAlignmentProblem, Patch
from local2global_embedding.run.utils import ScriptParser


def main(patch_folder: str, basename: str, dim: int):
    patch_list = []
    patch_folder = Path(patch_folder)
    patch_graph = torch.load(patch_folder / 'patch_graph.pt', map_location='cpu')

    for i in range(patch_graph.num_nodes):
        patch_file = patch_folder / f'patch{i}_data.pt'
        patch = torch.load(patch_file, map_location='cpu')
        coords = torch.load(patch_folder / f'{basename}_patch{i}_d{dim}_best_coords.pt', map_location='cpu')
        patch_list.append(Patch(patch.nodes, coords))

    prob = WeightedAlignmentProblem(patch_list, patch_edges=patch_graph.edges())
    patched_embedding_file = patch_folder / f'{basename}_d{dim}_coords.pt'
    patched_embedding_file_nt = patch_folder / f'{basename}_d{dim}_ntcoords.pt'
    ntcoords = prob.mean_embedding()
    coords = prob.get_aligned_embedding()
    torch.save(coords, patched_embedding_file)
    torch.save(ntcoords, patched_embedding_file_nt)


if __name__ == '__main__':
    parser = ScriptParser(main)
    args, kwargs = parser.parse()
    main(*args, **kwargs)