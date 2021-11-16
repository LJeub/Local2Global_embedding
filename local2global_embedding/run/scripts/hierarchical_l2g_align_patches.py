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
from shutil import copyfile, move
from tempfile import gettempdir, NamedTemporaryFile
from copy import copy

import numpy as np
import torch
from dask import delayed, bag
from dask.distributed import worker_client, secede, rejoin

from local2global.utils import WeightedAlignmentProblem, SVDAlignmentProblem
from local2global.patch import MeanAggregatorPatch
from local2global_embedding.clustering import spread_clustering, Partition
from local2global_embedding.sparsify import resistance_sparsify

from .utils import load_patches, move_to_tmp, restore_from_tmp, mean_embedding
from local2global_embedding.run.utils import ScriptParser


@delayed
def aligned_coords(patches, patch_graph, verbose=True, use_tmp=False):
    if use_tmp:
        patches = [move_to_tmp(p) for p in patches]
    else:
        patches = [copy(p) for p in patches]

    prob = SVDAlignmentProblem(patches, patch_graph.edges(), copy_data=False, verbose=verbose)
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

    if use_tmp:
        patches = [restore_from_tmp(p) for p in prob.patches]
    else:
        patches = prob.patches

    return MeanAggregatorPatch(patches)


def get_aligned_embedding(patch_graph, patches, clusters, verbose=True, use_tmp=False, resparsify=0):
    if not clusters:
        return aligned_coords(patches, patch_graph, verbose, use_tmp)
    else:
        cluster = clusters[0]
        reduced_patch_graph = patch_graph.partition_graph(cluster)
        if resparsify > 0:
            reduced_patch_graph = resistance_sparsify(reduced_patch_graph, resparsify)
        parts = Partition(cluster)
        reduced_patches = []
        for i, part in enumerate(parts):
            local_patch_graph = patch_graph.subgraph(part)
            local_patches = [patches[p] for p in part]
            reduced_patches.append(aligned_coords(
                patch_graph=local_patch_graph,
                patches=local_patches,
                verbose=verbose,
                use_tmp=use_tmp)
            )
        return get_aligned_embedding(reduced_patch_graph, reduced_patches, clusters[1:], verbose, use_tmp, resparsify)


def hierarchical_l2g_align_patches(patch_graph, shape, patches, output_file, cluster_file, mmap=False,
                                   verbose=False, use_tmp=False, resparsify=0):
    clusters = torch.load(cluster_file)
    if isinstance(clusters, list) and len(clusters) > 1:
        aligned = get_aligned_embedding(
                patch_graph=patch_graph, patches=patches, clusters=clusters[1:], verbose=verbose, use_tmp=use_tmp,
                resparsify=resparsify).coordinates
    else:
        aligned = aligned_coords(patches, patch_graph, verbose, use_tmp).coordinates

    if mmap:
        mean_embedding(aligned.patches, shape, output_file, use_tmp)
    else:
        coords = aligned.compute()
        np.save(output_file, np.asarray(coords, dtype=np.float32))

    return output_file


if __name__ == '__main__':
    ScriptParser(hierarchical_l2g_align_patches).run()
