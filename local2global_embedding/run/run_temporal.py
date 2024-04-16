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
from local2global_embedding.run.utils import load_data, ScriptParser, watch_progress


def run(name='LANL', data_root=None, data_opts={'protocol': 'TCP'}, dims=(2,),
        output='.', scale=True, alignment_type='temporal', alignment_window=14, use_median=True, cluster_init=True):
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
        patch_folder_name = '_'.join([name, model] + [f'{key}={value}' for key, value in data_opts.items()])

        patch_folder = output / patch_folder_name
        patch_folder.mkdir(parents=True, exist_ok=True)
        patches_s = []
        patches_t = []
        all_tasks = []
        for d in dims:
            for index in range(n_patches):
                patches = delayed(func.svd_patches)(data=data, index=index, output_folder=patch_folder, dim=d).persist()
                patches_s.append(patches[0].persist())
                patches_t.append(patches[1].persist())
            if alignment_type == 'temporal':
                error_s = client.submit(func.temporal_align_errors, patches=patches_s,
                                        scale=scale,
                                        output_file=patch_folder / f'source_temporal_alignment_errors.npy')
                all_tasks.append(error_s)

                error_t = client.submit(func.temporal_align_errors, patches=patches_t,
                                        scale=scale,
                                        output_file=patch_folder / f'dest_temporal_alignment_errors.npy')
                all_tasks.append(error_t)
            elif alignment_type == 'windowed':
                error_s = client.submit(func.windowed_align_errors, patches=patches_s,
                                        window=alignment_window, scale=scale, use_median=use_median,
                                        output_file=patch_folder / f'source_alignment_errors_window={alignment_window}.npy')
                all_tasks.append(error_s)

                error_t = client.submit(func.windowed_align_errors, patches=patches_t,
                                        window=alignment_window, scale=scale, use_median=use_median,
                                        output_file=patch_folder / f'dest_alignment_errors_window={alignment_window}.npy')
                all_tasks.append(error_t)
            elif alignment_type == 'global':
                error_s = client.submit(func.global_align_errors, patches=patches_s, window=alignment_window, scale=scale,
                                         output_file=patch_folder / f'source_global_alignment_errors_window={alignment_window}.npy')
                all_tasks.append(error_s)

                error_t = client.submit(func.global_align_errors, patches=patches_t, window=alignment_window, scale=scale,
                                         output_file=patch_folder / f'dest_global_alignment_errors_window={alignment_window}.npy')
                all_tasks.append(error_t)

        all_tasks.append(client.submit(func.leave_out_z_score_errors, error_file=error_s))
        all_tasks.append(client.submit(func.leave_out_z_score_errors, error_file=error_t))
        del error_t
        del error_s

        all_tasks = as_completed(all_tasks)
        watch_progress(all_tasks)


if __name__ == '__main__':
    ScriptParser(run).run()
