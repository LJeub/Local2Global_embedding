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
from numpy.lib.format import open_memmap
from dask.distributed import get_client, secede, rejoin

from local2global.utils import SVDAlignmentProblem, MeanAggregatorPatch
from local2global_embedding.clustering import metis_clustering
from local2global_embedding.patches import Partition

from .utils import load_patches
from local2global_embedding.run.utils import ScriptParser

def get_aligned_embedding(patch_graph, patches, levels, verbose=True):
    if levels == 1:
        prob = SVDAlignmentProblem(patches, patch_graph.edges(), copy_data=True, verbose=verbose)
        prob.align_patches(scale=False)
        return MeanAggregatorPatch(prob.patches)
    else:
        num_clusters = int(patch_graph.num_nodes ** (1 / levels))
        clusters = metis_clustering(patch_graph, num_clusters)
        reduced_patch_graph = patch_graph.partition_graph(clusters)
        parts = Partition(clusters)
        reduced_patches = []
        client = get_client()
        for i, part in enumerate(parts):
            local_patch_graph = patch_graph.subgraph(part)
            local_patches = [patches[p] for p in part]
            reduced_patches.append(
                client.submit(get_aligned_embedding,
                              patch_graph=local_patch_graph,
                              patches=local_patches,
                              levels=levels-1,
                              verbose=verbose)
            )
        secede()
        reduced_patches = client.gather(reduced_patches)
        rejoin()
        print('initialising alignment problem')
        prob = SVDAlignmentProblem(reduced_patches, patch_edges=reduced_patch_graph.edges(), copy_data=False,
                                   verbose=verbose)
        return MeanAggregatorPatch(prob.align_patches(scale=False).patches)


def hierarchical_l2g_align_patches(patch_graph, patch_folder: str, basename: str, dim: int, criterion: str, mmap=False,
                                   verbose=False, levels=1, output_file=None, use_tmp=False):
    patch_folder = Path(patch_folder)
    if output_file is None:
        if levels == 1:
            output_file = patch_folder / f'{basename}_d{dim}_l2g_{criterion}_coords.npy'
        else:
            output_file = patch_folder / f'{basename}_d{dim}_l2g_{criterion}_hc{levels}_coords.npy'

    patches = load_patches(patch_graph, patch_folder, basename, dim, criterion, lazy=mmap)

    aligned = get_aligned_embedding(patch_graph, patches, levels=levels, verbose=verbose)
    out = open_memmap(output_file, shape=aligned.shape, dtype=np.float32, mode='w+')
    out = aligned.coordinates.as_array(out)
    out.flush()
    return output_file

if __name__ == '__main__':
    ScriptParser(hierarchical_l2g_align_patches).run()
