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
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from filelock import SoftFileLock
from tqdm.auto import tqdm

from local2global_embedding.patches import create_patch_data
from local2global_embedding.run.utils import ScriptParser, patch_folder_name, load_data
from local2global_embedding.clustering import louvain_clustering, metis_clustering, distributed_clustering, fennel_clustering


def save_patch_data(graph, patch, filename):
    patch_graph = graph.subgraph(patch, relabel=False)
    torch.save(patch_graph, filename)


def prepare_patches(output_folder, name: str, min_overlap: int, target_overlap: int, data_root='/tmp',
                    min_patch_size: int = None, cluster='metis', num_clusters=10, num_iters: Optional[int]=None, beta=0.1,
                    sparsify='resistance', target_patch_degree=4.0, gamma=0.0, normalise=False, restrict_lcc=False,
                    verbose=False, use_tmp=False, mmap_edges: Optional[str] = None, mmap_features: Optional[str] = None):
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
        cluster_string = 'louvain'
    elif cluster == 'distributed':
        cluster_fun = lambda graph: distributed_clustering(graph, beta, rounds=num_iters)
        cluster_string = f'distributed_beta{beta}_it{num_iters}'
    elif cluster == 'fennel':
        cluster_fun = lambda graph: fennel_clustering(graph, num_clusters=num_clusters, num_iters=num_iters)
        cluster_string = f"fennel_n{num_clusters}_it{num_iters}"
    elif cluster == 'metis':
        cluster_fun = lambda graph: metis_clustering(graph, num_clusters=num_clusters)
        cluster_string = f"metis_n{num_clusters}"
    else:
        raise RuntimeError(f"Unknown cluster method '{cluster}'.")

    patch_folder = output_folder / patch_folder_name(name, min_overlap, target_overlap, cluster, num_clusters,
                                                     num_iters, beta, sparsify, target_patch_degree,
                                                     gamma)

    def load_graph():
        graph = load_data(name, root=data_root, mmap_edges=mmap_edges, mmap_features=mmap_features,
                          normalise=normalise, restrict_lcc=restrict_lcc)
        buffer_x = None
        buffer_e = None
        if use_tmp:
            if isinstance(graph.edge_index, np.memmap):
                buffer_e = TemporaryFile()
                edge_index = np.memmap(buffer_e, dtype=graph.edge_index.dtype, shape=graph.edge_index.shape)
                edge_index[:] = graph.edge_index
                graph.edge_index = edge_index
            if isinstance(graph.x, np.memmap):
                buffer_x = TemporaryFile()
                x = np.memmap(buffer_x, dtype=graph.x.dtype, shape=graph.x.shape)
                x[:] = graph.x
                graph.x = x
        if isinstance(graph.edge_index, np.memmap):
            graph.edge_index._mmap.madvise(mmap.MADV_RANDOM)
        if isinstance(graph.x, np.memmap):
            graph.x._mmap.madvise(mmap.MADV_RANDOM)
        return graph, buffer_x, buffer_e

    with SoftFileLock(patch_folder.with_suffix('.lock'), timeout=10):
        buffer_x = None
        buffer_e = None
        graph = None
        try:# make sure not to create patches twice
            if not (patch_folder / 'patch_graph.pt').is_file():
                print(f'creating patches in {patch_folder}')

                graph, buffer_x, buffer_e = load_graph()

                if isinstance(graph.edge_index, np.memmap):
                    print('using memory-mapped edge index')

                if isinstance(graph.x, np.memmap):
                    print('using memory-mapped features')

                cluster_file = output_folder / f"{name}_{cluster_string}_clusters.pt"
                if cluster_file.is_file():
                    clusters = torch.load(cluster_file, map_location='cpu')
                else:
                    clusters = cluster_fun(graph)
                    torch.save(clusters, cluster_file)

                patches, patch_graph = create_patch_data(graph, clusters, min_overlap, target_overlap, min_patch_size,
                                                         sparsify, target_patch_degree, gamma, verbose)
                patch_folder.mkdir(parents=True, exist_ok=True)

                print('saving patch index')
                for i, patch in tqdm(enumerate(patches)):
                    np.save(patch_folder / f'patch{i}_index.npy', patch)
                torch.save(patch_graph, patch_folder / 'patch_graph.pt')

                print("saving patch data")
                # with ThreadPoolExecutor() as executor:
                #     executor.map(save_patch_data, repeat(graph), patches, (patch_folder / f'patch{i}_data.pt' for i in len(patches)))
                for i, patch in tqdm(enumerate(patches)):
                    save_patch_data(graph, patch, patch_folder / f'patch{i}_data.pt')

            else:
                patch_graph = torch.load(patch_folder / 'patch_graph.pt')
                print('checking patch data')
                for i in tqdm(range(patch_graph.num_nodes)):
                    if not (patch_folder / f'patch{i}_data.pt').is_file():
                        if graph is None:
                            graph, buffer_x, buffer_e = load_graph()
                        print(f'saving missing patch {i}')
                        patch = np.load(patch_folder / f'patch{i}_index.npy')
                        save_patch_data(graph, patch, patch_folder / f'patch{i}_data.pt')
        finally:
            if buffer_e is not None:
                buffer_e.close()
            if buffer_x is not None:
                buffer_x.close()


if __name__ == '__main__':
    parser = ScriptParser(prepare_patches)
    args, kwargs = parser.parse()
    prepare_patches(**kwargs)
