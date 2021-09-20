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
import numpy as np
from numpy.lib.format import open_memmap
from filelock import SoftFileLock
import tqdm

from local2global import WeightedAlignmentProblem, Patch
from local2global_embedding.run.utils import ScriptParser


criterions = ['auc', 'loss']


def main(patch_folder: str, basename: str, dim: int, mmap=False):
    print(f'computing aligned embedding for {patch_folder}/{basename}_d{dim}')
    patch_list = []
    patch_folder = Path(patch_folder)
    patch_graph = torch.load(patch_folder / 'patch_graph.pt', map_location='cpu')

    mmap_load = 'r' if mmap else None
    for criterion in criterions:
        with SoftFileLock(patch_folder / f'{basename}_d{dim}_{criterion}_coords.lock', timeout=10):  # only one task at a time
            print('loading patch data')
            for i in tqdm(range(patch_graph.num_nodes)):
                patch_file = patch_folder / f'patch{i}_data.pt'
                patch = torch.load(patch_file, map_location='cpu')
                with SoftFileLock(f'{basename}_patch{i}_info.lock', timeout=10):
                    coords = np.load(patch_folder / f'{basename}_patch{i}_d{dim}_best_{criterion}_coords.npy',
                                     mmap_mode=mmap_load)
                patch_list.append(Patch(patch.nodes, coords))

            print('initialising alignment problem')
            prob = WeightedAlignmentProblem(patch_list, patch_edges=patch_graph.edges(), copy_data=False)
            patched_embedding_file = patch_folder / f'{basename}_d{dim}_{criterion}_coords.npy'
            patched_embedding_file_nt = patch_folder / f'{basename}_d{dim}_{criterion}_ntcoords.npy'
            if mmap:
                print('computing ntcoords using mmap')
                out = open_memmap(patched_embedding_file_nt, mode='w+', shape=(prob.n_nodes, prob.dim),
                                  dtype=np.float32)
                prob.mean_embedding(out)
                out.flush()
            else:
                print('computing ntcoords')
                out = np.empty(shape=(prob.n_nodes, prob.dim), dtype=np.float32)
                ntcoords = prob.mean_embedding(out)
                np.save(patched_embedding_file_nt, ntcoords)

            if mmap:
                print('computing aligned coords using mmap')
                out = open_memmap(patched_embedding_file, mode='w+', shape=(prob.n_nodes, prob.dim), dtype=np.float32)
                prob.align_patches().mean_embedding(out)
                out.flush()
            else:
                print('computing aligned coords')
                out = np.empty(shape=(prob.n_nodes, prob.dim), dtype=np.float32)
                coords = prob.align_patches().mean_embedding(out)
                np.save(patched_embedding_file, coords)


if __name__ == '__main__':
    ScriptParser(main).run()
