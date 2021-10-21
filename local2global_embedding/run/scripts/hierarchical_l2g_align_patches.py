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

import dask
import numpy as np
from numpy.lib.format import open_memmap
from dask.distributed import get_client, secede, rejoin, get_worker
from dask import delayed, compute

from local2global.utils import WeightedAlignmentProblem, MeanAggregatorPatch
from local2global_embedding.clustering import spread_clustering
from local2global_embedding.patches import Partition

from .utils import load_patches
from local2global_embedding.run.utils import ScriptParser


@delayed
def aligned_coords(patches, patch_graph, verbose=True):
    prob = WeightedAlignmentProblem(patches, patch_graph.edges(), copy_data=True, verbose=verbose)
    retry = True
    tries = 0
    max_tries = 3
    while retry and tries < max_tries:
        retry = False
        tries += 1
        try:
            prob.align_patches(scale=False)
        except Exception as e:
            print(e)
            if tries >= max_tries:
                raise e
            else:
                retry = True

    return MeanAggregatorPatch(prob.align_patches(scale=False).patches)


def get_aligned_embedding(patch_graph, patches, levels, verbose=True):
    if levels == 1:
        return aligned_coords(patches, patch_graph, verbose)
    else:
        num_clusters = int(patch_graph.num_nodes ** (1 / levels))
        clusters = spread_clustering(patch_graph, num_clusters)
        reduced_patch_graph = patch_graph.partition_graph(clusters)
        parts = Partition(clusters)
        reduced_patches = []
        for i, part in enumerate(parts):
            local_patch_graph = patch_graph.subgraph(part)
            local_patches = [patches[p] for p in part]
            reduced_patches.append(get_aligned_embedding(
                              patch_graph=local_patch_graph,
                              patches=local_patches,
                              levels=levels-1,
                              verbose=verbose)
            )
        return aligned_coords(reduced_patches, reduced_patch_graph, verbose)


def hierarchical_l2g_align_patches(patch_graph, patch_folder: str, basename: str, dim: int, criterion: str, mmap=False,
                                   verbose=False, levels=1, output_file=None, use_tmp=False):
    patch_folder = Path(patch_folder)
    if output_file is None:
        if levels == 1:
            output_file = patch_folder / f'{basename}_d{dim}_l2g_{criterion}_coords.npy'
        else:
            output_file = patch_folder / f'{basename}_d{dim}_l2g_{criterion}_hc{levels}_coords.npy'

    patches = load_patches(patch_graph, patch_folder, basename, dim, criterion, lazy=mmap)
    aligned = get_aligned_embedding(
                            patch_graph=patch_graph, patches=patches, levels=levels, verbose=verbose)
    aligned = aligned.compute()
    out = open_memmap(output_file, shape=aligned.shape, dtype=np.float32, mode='w+')
    out = aligned.coordinates.as_array(out)
    out.flush()
    return output_file


if __name__ == '__main__':
    ScriptParser(hierarchical_l2g_align_patches).run()
