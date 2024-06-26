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
from typing import Optional
from shutil import copyfile
from tempfile import NamedTemporaryFile, TemporaryFile
from local2global_embedding.utils import Timer

import torch
import numpy as np
from numpy.lib.format import open_memmap
from filelock import SoftFileLock
from tqdm import tqdm

from local2global.utils import WeightedAlignmentProblem, SVDAlignmentProblem
from local2global.patch import Patch, FilePatch
from local2global_embedding.run.utils import ScriptParser


def l2g_align_patches(patch_folder: str, basename: str, dim: int, criterion: str, mmap=False, use_tmp=False,
                      verbose=False):
    print(f'computing aligned embedding for {patch_folder}/{basename}_d{dim}')

    patch_folder = Path(patch_folder)
    patch_graph = torch.load(patch_folder / 'patch_graph.pt', map_location='cpu')

    with SoftFileLock(patch_folder / f'{basename}_d{dim}_{criterion}_coords.lock', timeout=10):  # only one task at a time
        patch_list = []
        for i in tqdm(range(patch_graph.num_nodes), desc='loading patch data', smoothing=0):
            node_file = patch_folder / f'patch{i}_index.npy'
            coords_file = patch_folder / f'{basename}_patch{i}_d{dim}_best_{criterion}_coords.npy'
            if node_file.is_file():
                nodes = np.load(node_file)
            else:
                patch_file = patch_folder / f'patch{i}_data.pt'
                patch = torch.load(patch_file, map_location='cpu')
                nodes = patch.nodes
            if mmap:
                if use_tmp:
                    coords_file_tmp = NamedTemporaryFile(delete=False)
                    coords_file_tmp.close()
                    copyfile(coords_file, coords_file_tmp.name)
                    coords_file = coords_file_tmp.name
                patch_list.append(FilePatch(nodes, coords_file))
            else:
                with SoftFileLock(f'{basename}_patch{i}_info.lock', timeout=10):
                    coords = np.load(coords_file)
                patch_list.append(Patch(nodes, coords))

        print('initialising alignment problem')
        timer = Timer()
        with timer:
            prob = SVDAlignmentProblem(patch_list, patch_edges=patch_graph.edges(), copy_data=False, verbose=verbose)
        patched_embedding_file = patch_folder / f'{basename}_d{dim}_{criterion}_coords.npy'
        patched_embedding_file_nt = patch_folder / f'{basename}_d{dim}_{criterion}_ntcoords.npy'
        if mmap:
            print('computing ntcoords using mmap')
            if use_tmp:
                print('using tmp buffer')
                with TemporaryFile() as f:
                    buffer = np.memmap(f, dtype=np.float32, shape=(prob.n_nodes, prob.dim))
                    prob.mean_embedding(buffer)
                    np.save(patched_embedding_file_nt, buffer)
            else:
                out = open_memmap(patched_embedding_file_nt, mode='w+', shape=(prob.n_nodes, prob.dim),
                                  dtype=np.float32)
                prob.mean_embedding(out)
                out.flush()
        else:
            print('computing ntcoords')
            out = np.zeros(shape=(prob.n_nodes, prob.dim), dtype=np.float32)
            ntcoords = prob.mean_embedding(out)
            np.save(patched_embedding_file_nt, ntcoords)

        if mmap:
            print('computing aligned coords using mmap')
            if use_tmp:
                with TemporaryFile() as f:
                    buffer = np.memmap(f, dtype=np.float32, shape=(prob.n_nodes, prob.dim))
                    with timer:
                        prob.align_patches().mean_embedding(buffer)
                    np.save(patched_embedding_file, buffer)
            else:
                out = open_memmap(patched_embedding_file, mode='w+', shape=(prob.n_nodes, prob.dim), dtype=np.float32)
                with timer:
                    prob.align_patches().mean_embedding(out)
                out.flush()
        else:
            print('computing aligned coords')
            out = np.empty(shape=(prob.n_nodes, prob.dim), dtype=np.float32)
            with timer:
                coords = prob.align_patches().mean_embedding(out)
            np.save(patched_embedding_file, coords)
        with open(patched_embedding_file.with_name(patched_embedding_file.stem + "timing.txt")) as f:
            f.write(timer.total)


if __name__ == '__main__':
    ScriptParser(l2g_align_patches).run()
