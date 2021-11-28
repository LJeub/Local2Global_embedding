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
from dask.distributed import Client, LocalCluster, as_completed
from dask import delayed
from runpy import run_path

from local2global_embedding.run.scripts import functions as func
from local2global_embedding.patches import rolling_window_graph
from local2global_embedding.run.utils import load_data, ScriptParser, watch_progress


def run(name='LANL', data_root=None, data_opts={'protocol': 'TCP'}, dims=(2,),
        output='.', scale=True, verbose_l2g=False, levels=1, window=7, cluster_init=True):
    model = 'SVD'
    cluster_init_path = Path().home() / '.config' / 'dask' / 'cluster_init.py'
    if cluster_init and cluster_init_path.is_file():
        kwargs = run_path(cluster_init_path)
        cluster = kwargs['cluster']
    else:
        cluster = LocalCluster()
    with Client(cluster) as client:
        print(client.dashboard_link)
        output = Path(output)
        data = load_data(name, root=data_root, **data_opts)
        n_patches = len(data.timesteps)
        patch_graph = delayed(rolling_window_graph)(n_patches, window)
        patch_folder_name = '_'.join([name, model] + [f'{key}={value}' for key, value in data_opts.items()])

        patch_folder = output / patch_folder_name
        patch_folder.mkdir(parents=True, exist_ok=True)
        patches_s = []
        patches_t = []
        all_tasks = []
        for d in dims:
            for index in range(n_patches):
                patches = delayed(func.svd_patches)(data=data, index=index, output_folder=patch_folder, dim=d)
                patches_s.append(patches[0])
                patches_t.append(patches[1])

            all_tasks.append(
                client.submit(
                    func.hierarchical_l2g_align_patches,
                        patch_graph=patch_graph, shape=(n_patches, d), patches=patches_s,
                        output_file=patch_folder / f'source_{model}_{d}_mean_coords.npy',
                        cluster_file=None, mmap=False,
                        verbose=verbose_l2g, use_tmp=False, resparsify=0,
                        store_aligned_patches=True, scale=scale))

            all_tasks.append(
                client.submit(func.hierarchical_l2g_align_patches,
                        patch_graph=patch_graph, shape=(n_patches, d), patches=patches_t,
                        output_file=patch_folder / f'dest_{model}_{d}_mean_coords.npy',
                        cluster_file=None, mmap=False,
                        verbose=verbose_l2g, use_tmp=False, resparsify=0,
                        store_aligned_patches=True, scale=scale))
        all_tasks = as_completed(all_tasks)
        watch_progress(all_tasks)


if __name__ == '__main__':
    ScriptParser(run).run()
