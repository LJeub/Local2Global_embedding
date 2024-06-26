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
from tempfile import TemporaryFile
import mmap
import sys

import numpy as np
import torch
from filelock import SoftFileLock
from tqdm.auto import tqdm

from local2global_embedding.patches import create_patch_data, merge_small_clusters
from local2global_embedding.network import TGraph
from local2global_embedding.run.utils import ScriptParser, patch_folder_name, cluster_file_name, load_data
from local2global_embedding.clustering import louvain_clustering, metis_clustering, distributed_clustering, \
    fennel_clustering, hierarchical_aglomerative_clustering
from local2global_embedding.utils import Timer


def prepare_patches(output_folder, name: str, min_overlap: int, target_overlap: int, graph,
                    min_patch_size: int = None, cluster='metis', num_clusters=10, num_iters: Optional[int]=None,
                    beta=0.1, levels=1,
                    sparsify='resistance', target_patch_degree=4.0, gamma=0.0,
                    verbose=False):
    """
    initialise patch data

    Args:
        output_folder: experiment folder
        name: name of data set
        data_root: root dir for data set
        min_overlap: minimum patch overlap
        target_overlap: desired patch overlap
        min_patch_size: minimum patch size
        cluster: cluster method (one of {'metis', 'louvain', 'distributed', 'fennel'}
        num_clusters: number of clusters for metis/fennel
        num_iters: number of iterations for fennel/distributed
        beta: beta value for distributed
        sparsify: sparsification method (one of {'resistance', 'rmst', 'none'})
        target_patch_degree: target patch degree for resistance sparsification
        gamma: gamma value for rmst sparsification
        verbose: print output
    """

    output_folder = Path(output_folder)

    if cluster == 'louvain':
        cluster_fun = lambda graph: louvain_clustering(graph)
    elif cluster == 'distributed':
        cluster_fun = lambda graph: distributed_clustering(graph, beta, rounds=num_iters)
    elif cluster == 'fennel':
        cluster_fun = lambda graph: fennel_clustering(graph, num_clusters=num_clusters, num_iters=num_iters)
    elif cluster == 'metis':
        cluster_fun = lambda graph: metis_clustering(graph, num_clusters=num_clusters)
    else:
        raise RuntimeError(f"Unknown cluster method '{cluster}'.")

    patch_folder = output_folder / patch_folder_name(name, min_overlap, target_overlap, cluster, num_clusters,
                                                     num_iters, beta, levels, sparsify, target_patch_degree,
                                                     gamma)
    cluster_file = output_folder / cluster_file_name(name, cluster, num_clusters, num_iters, beta, levels)

    with SoftFileLock(patch_folder.with_suffix('.lock')):
        if isinstance(graph.edge_index, np.memmap):
            graph.edge_index._mmap.madvise(mmap.MADV_RANDOM)
        if isinstance(graph.x, np.memmap):
            graph.x._mmap.madvise(mmap.MADV_RANDOM)

        if not (patch_folder / 'patch_graph.pt').is_file():
            print(f'creating patches in {patch_folder}')
            patch_folder.mkdir(parents=True, exist_ok=True)

            if cluster_file.is_file():
                clusters = torch.load(cluster_file, map_location='cpu')
            else:
                cl_timer = Timer()
                with cl_timer:
                    clusters = cluster_fun(graph)
                    if levels > 1:
                        clusters = [merge_small_clusters(graph, clusters, min_overlap)]
                        clusters.extend(hierarchical_aglomerative_clustering(graph.partition_graph(clusters[0]),
                                                                             levels=levels-1))
                torch.save(clusters, cluster_file)
                with open(cluster_file.with_name(f"{cluster_file.stem}_timing.txt"), 'w') as f:
                    f.write(str(cl_timer.total))

            pc_timer = Timer()
            with pc_timer:
                patches, patch_graph = create_patch_data(graph, clusters, min_overlap, target_overlap, min_patch_size,
                                                         sparsify, target_patch_degree, gamma, verbose)

            for i, patch in tqdm(enumerate(patches), total=len(patches), desc='saving patch index'):
                np.save(patch_folder / f'patch{i}_index.npy', patch)
            with open(patch_folder / "patch_graph_creation_time.txt", "w") as f:
                f.write(str(pc_timer.total))
            torch.save(patch_graph, patch_folder / 'patch_graph.pt')
        else:
            patch_graph = torch.load(patch_folder / 'patch_graph.pt')
    return patch_graph


if __name__ == '__main__':
    parser = ScriptParser(prepare_patches)
    args, kwargs = parser.parse()
    prepare_patches(**kwargs)
