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

"""Training run script"""

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
import sys
from pathlib import Path
from typing import List
from runpy import run_path

import torch
import dask
import dask.distributed
import enlighten
from dask.distributed import as_completed, Client, get_worker
import logging

from local2global_embedding.run.utils import ResultsDict, load_data, ScriptParser, patch_folder_name
from local2global_embedding.run.scripts import functions as func


def with_dependencies(f):
    def f_d(*args, _depends_on=None, **kwargs):
        return f(*args, **kwargs)

    f_d.__name__ = f.__name__
    return f_d


def run(name='Cora', data_root='/tmp', no_features=False, model='VGAE', num_epochs=10000,
        patience=20, runs=10, cl_runs=50, dims: List[int] = None, hidden_multiplier=2, target_patch_degree=4.0,
        min_overlap: int = None, target_overlap: int = None, gamma=0.0, sparsify='resistance',
        cluster='metis', num_clusters=10, beta=0.1, num_iters: int = None, lr=0.001, cl_lr=0.01, dist=False,
        output='.', device: str = None, verbose=False, levels=1,
        run_baseline=True, normalise=False, restrict_lcc=False, mmap_edges=False, mmap_features=False,
        random_split=False, use_tmp=False, cluster_init=False, use_gpu_frac=1.0):
    """
    Run training example.

    By default this function writes results to the current working directory. To override this use the ``output``
    argument.

    This function reproduces figure 1(a) of [#l2g]_ if called as ``run(dims=[2**i for i in range(1, 8)], plot=True)``.

    Args:
        name: Name of data set to load (one of {``'Cora'``, ``'PubMed'``, ``'AMZ_computers'``, ``'AMZ_photo'``})
        data_root: Directory to use for downloaded data
        no_features: If ``True``, discard features and use node identity.
        model: embedding model type (one of {'VGAE', 'GAE', 'DGI'})
        num_epochs: Number of training epochs
        patience: Patience for early stopping
        runs: Number of training runs (keep best result)
        dims: list of embedding dimensions (default: ``[2]``)
        hidden_multiplier: Hidden dimension is ``hidden_multiplier * dim``
        target_patch_degree: Target patch degree for resistance sparsification.
        min_overlap: Minimum target patch overlap (default: ``max(dims) + 1``)
        target_overlap: Target patch overlap (default: ``2 * max(dims)``)
        gamma: Value of 'gamma' for RMST sparsification
        sparsify: Sparsification method to use (one of {``'resistance'``, ``'none'``, ``'rmst'``})
        cluster: Clustering method to use (one of {``'louvain'``, ``'fennel'`` , ``'distributed'``, ``'metis'``})
        num_clusters: Target number of clusters for distributed, fennel, or metis.
        num_iters: Maximum iterations for distributed or fennel
        lr: Learning rate
        dist: If ``True``, use distance decoder instead of inner product decoder
        output: output folder
        device: Device used for training e.g., 'cpu', 'cuda' (defaults to 'cuda' if available else 'cpu')
        plot: If ``True``, plot embedding performance
        verbose: If ``True``, show progress info
        max_workers: maximum number of parallel training jobs
        cmd_prefix: prefix used for submitting jobs (e.g. srun for slurm clusters)
        run_baseline: if ``True``, run baseline full model

    This function is also exposed as a command-line interface.

    .. rubric:: References

    .. [#l2g] L. G. S. Jeub et al.
              “Local2Global: Scaling global representation learning on graphs via local training”.
              DLG-KDD’21. 2021. `arXiv:2107.12224 [cs.LG] <https://arxiv.org/abs/2107.12224>`_.

    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    if cluster_init:
        cluster_init_path = Path().home() / '.config' / 'dask' / 'cluster_init.py'
        kwargs = run_path(cluster_init_path)
        client = Client(kwargs['cluster'])
    else:
        client = Client()
    print(client.dashboard_link)
    if 'gpu' in dask.config.get('distributed.worker.resources'):
        gpu_req = {'gpu': use_gpu_frac}
    else:
        gpu_req = {}

    if dims is None:
        dims = [2]
    output_folder = Path(output).expanduser()
    data_root = Path(data_root).expanduser()
    print(f'Started experiment for data set {name}')
    print(f'Results will be placed in {output_folder.resolve()}')
    print(f'Data root is {data_root.resolve()}')
    if use_tmp:
        print('Any memmapped data will be moved to local storage.')
    if normalise:
        print('features will be normalised before training')

    manager = enlighten.get_manager(threaded=True)
    patch_create_progress = manager.counter(desc='create patches', total=0, file=sys.stdout)
    baseline_progress = manager.counter(desc='baseline', total=0, file=sys.stdout)
    patch_progress = manager.counter(desc='patch', total=0, file=sys.stdout)
    align_progress = manager.counter(desc='align', total=0, file=sys.stdout)
    eval_progress = manager.counter(desc='eval', total=0, file=sys.stdout)
    total_progress = manager.counter(desc='total', total=0, file=sys.stdout)

    def progress_callback(bar):
        bar.total += 1
        total_progress.total += 1

        def callback(future):
            bar.update()
            total_progress.update()

        return callback

    mmap_edges = 'r' if mmap_edges else None
    mmap_features = 'r' if mmap_features else None

    train_basename = f'{name}_{model}'
    eval_basename = f'{name}_{model}'
    min_overlap = min_overlap if min_overlap is not None else max(dims) + 1
    target_overlap = target_overlap if target_overlap is not None else 2 * max(dims)

    if dist:
        eval_basename += '_dist'
        if model != 'DGI':
            train_basename += '_dist'

    if normalise:
        eval_basename += '_norm'
        train_basename += '_norm'

    all_tasks = as_completed()

    patch_graph_remote = client.submit(func.prepare_patches, pure=False,
                                       output_folder=output_folder, name=name, data_root=data_root,
                                       min_overlap=min_overlap, target_overlap=target_overlap,
                                       cluster=cluster,
                                       num_clusters=num_clusters, num_iters=num_iters, beta=beta,
                                       sparsify=sparsify, target_patch_degree=target_patch_degree,
                                       gamma=gamma,
                                       verbose=False,
                                       normalise=normalise,
                                       restrict_lcc=restrict_lcc, use_tmp=use_tmp,
                                       mmap_edges=mmap_edges,
                                       mmap_features=mmap_features)
    patch_graph_remote.add_done_callback(progress_callback(patch_create_progress))
    all_tasks.add(patch_graph_remote)

    if run_baseline:
        # compute baseline full model if necessary
        baseline_info_file = output_folder / f'{train_basename}_full_info.json'
        baseline_loss_eval_file = output_folder / f'{eval_basename}_full_loss_eval.json'
        baseline_auc_eval_file = baseline_loss_eval_file.with_name(
            baseline_loss_eval_file.name.replace('_loss_', '_auc_'))
        data_file = output_folder / f'{name}_data.pt'
        if not data_file.is_file():
            data = load_data(name, data_root, restrict_lcc=restrict_lcc, mmap_edges=mmap_edges,
                             mmap_features=mmap_features)
            torch.save(data, data_file)
        for d in dims:
            baseline_tasks = []
            with ResultsDict(baseline_info_file) as baseline_data:
                r = baseline_data.runs(d)

            if r < runs:
                task = client.submit(func.train, pure=False, resources=gpu_req,
                                     data=data_file, model=model,
                                     lr=lr, num_epochs=num_epochs,
                                     patience=patience, verbose=verbose,
                                     results_file=baseline_info_file, dim=d,
                                     hidden_multiplier=hidden_multiplier,
                                     no_features=no_features, dist=dist,
                                     device=device, runs=runs, normalise_features=normalise)
                task.add_done_callback(progress_callback(baseline_progress))
                baseline_tasks.append(task)
                all_tasks.add(task)
                del task

            with ResultsDict(baseline_loss_eval_file, replace=True) as eval_results:
                if baseline_tasks or not eval_results.contains_dim(d):
                    coords_file = output_folder / f'{train_basename}_full_d{d}_best_loss_coords.npy'
                    task = client.submit(with_dependencies(func.evaluate), pure=False, _depends_on=baseline_tasks,
                                         resources=gpu_req,
                                         name=name,
                                         data_root=data_root,
                                         restrict_lcc=restrict_lcc,
                                         embedding_file=coords_file,
                                         results_file=baseline_loss_eval_file,
                                         dist=dist,
                                         device=device,
                                         lr=cl_lr,
                                         runs=cl_runs,
                                         random_split=random_split,
                                         mmap_edges=mmap_edges,
                                         mmap_features=mmap_features)
                    task.add_done_callback(progress_callback(eval_progress))
                    all_tasks.add(task)
                    del task

            with ResultsDict(baseline_auc_eval_file, replace=True) as eval_results:
                if baseline_tasks or not eval_results.contains_dim(d):
                    coords_file = output_folder / f'{train_basename}_full_d{d}_best_auc_coords.npy'
                    task = client.submit(with_dependencies(func.evaluate), pure=False, _depends_on=baseline_tasks,
                                         resources=gpu_req,
                                         name=name,
                                         data_root=data_root,
                                         restrict_lcc=restrict_lcc,
                                         embedding_file=coords_file,
                                         results_file=baseline_auc_eval_file,
                                         dist=dist,
                                         device=device,
                                         lr=cl_lr,
                                         runs=cl_runs,
                                         random_split=random_split,
                                         mmap_edges=mmap_edges,
                                         mmap_features=mmap_features,
                                         )
                    task.add_done_callback(progress_callback(eval_progress))
                    all_tasks.add(task)
                    del task
            del baseline_tasks

    patch_folder = output_folder / patch_folder_name(name, min_overlap, target_overlap, cluster, num_clusters,
                                                     num_iters, beta, sparsify, target_patch_degree,
                                                     gamma)

    patch_graph = patch_graph_remote.result()
    for d in dims:
        patch_tasks = []
        for pi in range(patch_graph.num_nodes):
            patch_data_file = patch_folder / f'patch{pi}_data.pt'
            patch_result_file = patch_folder / f'{train_basename}_patch{pi}_info.json'
            with ResultsDict(patch_result_file) as patch_results:
                r = patch_results.runs(d)
            if r < runs:
                task = client.submit(func.train, pure=False, resources=gpu_req,
                                     data=patch_data_file, model=model,
                                     lr=lr, num_epochs=num_epochs,
                                     patience=patience, verbose=verbose,
                                     results_file=patch_result_file, dim=d,
                                     hidden_multiplier=hidden_multiplier,
                                     no_features=no_features, dist=dist,
                                     device=device, runs=runs, normalise_features=normalise)
                task.add_done_callback(progress_callback(patch_progress))
                patch_tasks.append(task)
                all_tasks.add(task)
                del task

        for criterion in ('auc', 'loss'):
            if levels == 1:
                l2g_coords_file = patch_folder / f'{train_basename}_d{d}_l2g_{criterion}_coords.npy'
                l2g_eval_file = patch_folder / f'{eval_basename}_l2g_{criterion}_eval.json'
            else:
                l2g_coords_file = patch_folder / f'{train_basename}_d{d}_l2g_hc{levels}_{criterion}_coords.npy'
                l2g_eval_file = patch_folder / f'{eval_basename}_l2g_hc{levels}_{criterion}_eval.json'
            nt_coords_file = patch_folder / f'{train_basename}_d{d}_nt_{criterion}_coords.npy'
            nt_eval_file = patch_folder / f'{eval_basename}_nt_{criterion}_eval.json'

            l2g_task = False
            if patch_tasks or not l2g_coords_file.is_file():
                l2g_task = client.submit(with_dependencies(func.hierarchical_l2g_align_patches), pure=False,
                                         _depends_on=patch_tasks,
                                         patch_graph=patch_graph_remote,
                                         patch_folder=patch_folder, basename=train_basename, dim=d,
                                         criterion=criterion,
                                         mmap=mmap_features is not None, use_tmp=use_tmp, verbose=verbose,
                                         levels=levels,
                                         output_file=l2g_coords_file)
                l2g_task.add_done_callback(progress_callback(align_progress))
                all_tasks.add(l2g_task)
            with ResultsDict(l2g_eval_file, replace=True) as l2g_eval:
                if l2g_task or not l2g_eval.contains_dim(d):
                    task = client.submit(with_dependencies(func.evaluate), pure=False, _depends_on=l2g_task,
                                         resources=gpu_req,
                                         name=name,
                                         data_root=data_root,
                                         restrict_lcc=restrict_lcc,
                                         embedding_file=l2g_coords_file,
                                         results_file=l2g_eval_file,
                                         dist=dist,
                                         device=device,
                                         lr=cl_lr,
                                         runs=cl_runs,
                                         random_split=random_split,
                                         mmap_edges=mmap_edges,
                                         mmap_features=mmap_features,
                                         )
                    task.add_done_callback(progress_callback(eval_progress))
                    all_tasks.add(task)
                    del task
            del l2g_task

            nt_task = False
            if patch_tasks or not nt_coords_file.is_file():
                nt_task = client.submit(with_dependencies(func.no_transform_embedding), pure=False,
                                        _depends_on=patch_tasks,
                                        patch_graph=patch_graph_remote,
                                        patch_folder=patch_folder,
                                        basename=train_basename,
                                        dim=d,
                                        criterion=criterion,
                                        mmap=mmap_features is not None,
                                        use_tmp=use_tmp
                                        )
                nt_task.add_done_callback(progress_callback(align_progress))
                all_tasks.add(nt_task)

            with ResultsDict(nt_eval_file, replace=True) as nt_eval:
                if nt_task or not nt_eval.contains_dim(d):
                    task = client.submit(with_dependencies(func.evaluate), pure=False, _depends_on=nt_task,
                                         resources=gpu_req,
                                         name=name,
                                         data_root=data_root,
                                         restrict_lcc=restrict_lcc,
                                         embedding_file=nt_coords_file,
                                         results_file=nt_eval_file,
                                         dist=dist,
                                         device=device,
                                         lr=cl_lr,
                                         runs=cl_runs,
                                         random_split=random_split,
                                         mmap_edges=mmap_edges,
                                         mmap_features=mmap_features,
                                         )
                    task.add_done_callback(progress_callback(eval_progress))
                    all_tasks.add(task)
                    del task
            del nt_task
        del patch_tasks

    baseline_progress.refresh()
    patch_progress.refresh()
    align_progress.refresh()
    eval_progress.refresh()
    total_progress.refresh()

    # make sure to wait for all tasks to complete and report overall progress
    for c in all_tasks:
        print(f'{c} complete')
        del c
    manager.stop()


if __name__ == '__main__':
    # run main script
    parser = ScriptParser(run).run()