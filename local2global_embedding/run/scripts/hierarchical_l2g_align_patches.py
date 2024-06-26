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

import numpy as np
import torch
from pathlib import Path

from dask import delayed
from dask.distributed import worker_client, Client
from distributed import secede, rejoin

from local2global_embedding.clustering import Partition
from local2global_embedding.sparsify import resistance_sparsify
from local2global_embedding.utils import Timer

from .utils import mean_embedding, aligned_coords
from local2global_embedding.run.utils import ScriptParser
from filelock import SoftFileLock


def get_aligned_embedding(patch_graph, patches, clusters, verbose=True, use_tmp=False, resparsify=0,
                          scale=False, rotate=True, translate=True, time=0.0):
    if not clusters:
        coords, ltime = aligned_coords(patches, patch_graph, verbose, use_tmp, scale, rotate, translate)
        return coords, time + ltime
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
            rpatch, rtime = aligned_coords(
                patch_graph=local_patch_graph,
                patches=local_patches,
                verbose=verbose,
                use_tmp=use_tmp)
            reduced_patches.append(rpatch)
            time += rtime
        return get_aligned_embedding(reduced_patch_graph, reduced_patches, clusters[1:], verbose, use_tmp, resparsify,
                                     scale, rotate, translate, time)





def hierarchical_l2g_align_patches(patch_graph, shape, patches, output_file: Path, cluster_file=None, mmap=False,
                                   verbose=False, use_tmp=False, resparsify=0, store_aligned_patches=False, scale=False,
                                   rotate=True, translate=True):

    if mmap:
        def get_coords(aligned):
            return mean_embedding(aligned.patches, shape, output_file, use_tmp)
    else:
        def get_coords(aligned):
            coords = np.asarray(aligned.coordinates, dtype=np.float32)
            np.save(output_file, np.asarray(coords, dtype=np.float32))
            return coords
    @delayed
    def save_results(aligned, time):
        coords = get_coords(aligned)
        if store_aligned_patches:
            if scale:
                postfix = '_aligned_scaled_coords'
            else:
                postfix = '_aligned_coords'
            for patch in aligned.patches:
                f_name = patch.coordinates.filename
                aligned_f_name = f_name.with_name(f_name.name.replace('_coords', postfix))
                np.save(aligned_f_name, patch.coordinates)

        timing_file = output_file.with_name(output_file.stem + "time.txt")
        with SoftFileLock(timing_file.with_suffix(".lock")):
            with open(timing_file, 'a') as f:
                f.write(str(time) + "\n")
        return coords

    if cluster_file is not None:
        clusters = torch.load(cluster_file)
    else:
        clusters = None
    if isinstance(clusters, list) and len(clusters) > 1:
        clusters = delayed(torch.load)(cluster_file)
        aligned, time = get_aligned_embedding(
                patch_graph=patch_graph, patches=patches, clusters=clusters[1:], verbose=verbose, use_tmp=use_tmp,
                resparsify=resparsify, scale=scale, rotate=rotate, translate=translate)
    else:
        aligned, time = aligned_coords(patches, patch_graph, verbose, use_tmp, scale, rotate, translate)
    return save_results(aligned, time)


if __name__ == '__main__':
    client = Client()
    ScriptParser(hierarchical_l2g_align_patches).run()
