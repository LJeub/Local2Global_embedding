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
import os
from filelock import SoftFileLock, FileLock

from local2global_embedding.network import TGraph
from local2global_embedding.run.once_per_worker import once_per_worker
from weakref import finalize

from local2global_embedding.run.once_per_worker.once_per_worker import OncePerWorker

print('importing build-in modules')
import sys
from pathlib import Path
from typing import List
from runpy import run_path
from traceback import print_exception
from collections.abc import Iterable
from copy import copy

print('importing numpy')
import numpy as np

print('importing pytorch')
import torch

print('importing dask')
import dask
import dask.distributed
from dask.distributed import as_completed, Client, get_worker

print('importing log and progress modules')
from tqdm import tqdm
import enlighten
import logging

print('importing needed functions')
from local2global_embedding.run.utils import (ResultsDict, load_data, ScriptParser, patch_folder_name,
                                              cluster_file_name, watch_progress,
                                              load_classification_problem)
from local2global_embedding.utils import speye, set_device
from local2global_embedding.run.scripts import functions as func
from local2global_embedding.run.scripts.utils import build_patch, ScopedTemporaryFile
from functools import partialmethod
from tempfile import NamedTemporaryFile


def with_dependencies(f):
    def f_d(*args, _depends_on=None, **kwargs):
        return f(*args, **kwargs)

    f_d.__name__ = f.__name__
    return f_d


def run(name='Cora', data_root='/tmp', no_features=False, model='VGAE', num_epochs=10000,
        patience=20, runs=10, cl_runs=5, dims: List[int] = None, hidden_multiplier=2, target_patch_degree=4.0,
        min_overlap: int = None, target_overlap: int = None, gamma=0.0, sparsify='resistance', train_directed=False,
        cluster='metis', num_clusters=10, beta=0.1, num_iters: int = None, lr=0.001, cl_model='logistic',
        cl_train_args={}, cl_model_args={}, dist=False,
        output='.', device: str = None, verbose_train=False, verbose_l2g=False, levels=1, resparsify=0,
        run_baseline=True, normalise=False, restrict_lcc=False, scale=False, rotate=True, translate=True,
        mmap_edges=False, mmap_features=False,
        random_split=True, use_tmp=False, cluster_init=False, use_gpu_frac=1.0, grid_search_params=True,
        progress_bars=True):
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
        verbose_train: If ``True``, show progress info
        max_workers: maximum number of parallel training jobs
        cmd_prefix: prefix used for submitting jobs (e.g. srun for slurm clusters)
        run_baseline: if ``True``, run baseline full model

    This function is also exposed as a command-line interface.

    .. rubric:: References

    .. [#l2g] L. G. S. Jeub et al.
              “Local2Global: Scaling global representation learning on graphs via local training”.
              DLG-KDD’21. 2021. `arXiv:2107.12224 [cs.LG] <https://arxiv.org/abs/2107.12224>`_.

    """
    if not progress_bars:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    grid_search_params = grid_search_params and cl_model == 'mlp'  # grid search only implemented for MLP
    if grid_search_params:
        eval_func = func.mlp_grid_search_eval
    else:
        eval_func = func.evaluate

    if verbose_train:
        logging.basicConfig(level=logging.DEBUG)

    if cluster_init:
        print('setting up cluster')
        cluster_init_path = Path().home() / '.config' / 'dask' / 'cluster_init.py'
        kwargs = run_path(cluster_init_path)
        client = Client(kwargs['cluster'])
    else:
        print('launching default client')
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
    result_folder_name = f'{num_epochs=}_{patience=}_{lr=}'
    print(f'Started experiment for data set {name}')
    print(f'Results will be placed in {output_folder.resolve()}')
    print(f'Data root is {data_root.resolve()}')
    if use_tmp:
        print('Any memmapped data will be moved to local storage.')
    if normalise:
        print('features will be normalised before training')

    manager = enlighten.get_manager(threaded=True)
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

    if train_directed:
        train_basename = f'{name}_dir_{model}'
        eval_basename = f'{name}_dir_{model}'
    else:
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

    if grid_search_params:
        eval_basename += '_gridsearch'
    eval_basename += f'_{cl_model}'
    if cl_model_args:
        eval_basename += f'({cl_model_args})'
    if cl_train_args:
        eval_basename += f'{cl_train_args}'

    l2g_name = 'l2g'
    if scale:
        l2g_name += '_scale'
    if not rotate:
        l2g_name += "_norotate"
    if not translate:
        l2g_name += "_notranslate"
    if levels > 1:
        l2g_name += f'_hc{levels}'

    if isinstance(lr, Iterable):
        lr = list(lr)
        if len(lr) < runs:
            if len(lr) == 2:
                lr = np.logspace(np.log10(lr[0]), np.log10(lr[1]), runs)
            else:
                raise ValueError(f'Number of learning rates {len(lr)} specified does not match number of runs {runs}.')
    else:
        lr = [lr for _ in range(runs)]

    all_tasks = as_completed()

    def build_training_data(data, device):
        device = set_device(device)
        data = data.to(TGraph).to(device=device)
        if no_features:
            data.x = speye(data.num_nodes).to(device)
        else:
            data.x = torch.as_tensor(data.x, dtype=torch.float32)
            if normalise:
                r_sum = data.x.sum(dim=1)
                r_sum[r_sum == 0] = 1.0  # avoid division by zero
                data.x /= r_sum[:, None]
        return data

    @dask.delayed(pure=True, traverse=False)
    def load_patch(patch_folder, data, i):
        data = data._get_value()
        nodes = np.load(patch_folder / f'patch{i}_index.npy')
        return OncePerWorker.instance_for_function(lambda: build_training_data(data.subgraph(nodes, relabel=False), device))

    def load_patch_data(patch_folder, data, n_patches):
        return [load_patch(patch_folder, data, i) for i in range(n_patches)]

    def load_and_copy_data():
        data = load_data(name=name, root=data_root, restrict_lcc=restrict_lcc,
                         mmap_edges=mmap_edges, mmap_features=mmap_features, directed=train_directed)
        if use_tmp:
            tmpdir = Path(os.getenv("TMPDIR", "/tmp"))
            e_file = tmpdir / f"{name}_edges.npy"
            x_file = tmpdir / f"{name}_x.npy"
            if isinstance(data.edge_index, np.memmap):
                with FileLock(tmpdir/f"{name}_edges.lock"):
                    if not (e_file).is_file():
                        np.save(e_file, data.edge_index)
                    data.edge_index = np.load(e_file, mmap_mode='r')

            if isinstance(data.x, np.memmap):
                with FileLock(tmpdir/f"{name}_x.lock"):
                    if not x_file.is_file():
                        np.save(x_file, data.x)
                    data.x = np.load(x_file, mmap_mode='r')

        cl_data = load_classification_problem(name, data, root=data_root)
        data.cl_data = cl_data
        return data

    data = None
    baseline_train_data = None

    def _load_data():
        nonlocal data
        if data is None:
            data = once_per_worker(load_and_copy_data)

    @dask.delayed(pure=True, traverse=False)
    def build_baseline_training_data(data):
        data = copy(data._get_value())
        return OncePerWorker.instance_for_function(lambda: build_training_data(data, device))

    def _baseline_data():
        nonlocal baseline_train_data
        _load_data()
        if baseline_train_data is None:
            baseline_train_data = build_baseline_training_data(data)

    if run_baseline:
        # compute baseline full model if necessary
        result_folder = output_folder / result_folder_name
        result_folder.mkdir(exist_ok=True)

        baseline_eval_file = result_folder / f'{eval_basename}_full_eval.json'

        for d in dims:
            with ResultsDict(baseline_eval_file, lock=False) as baseline_data:
                r_done = baseline_data.runs(d)
            coords = []
            for r in range(r_done, runs):
                _load_data()
                _baseline_data()
                baseline_info_file = result_folder / f'{train_basename}_d{d}_r{r}_full_info.json'
                coords_task = dask.delayed(func.train, pure=False)(
                    data=baseline_train_data, model=model,
                    lr=lr[r], num_epochs=num_epochs,
                    patience=patience, verbose=verbose_train,
                    results_file=baseline_info_file, dim=d,
                    hidden_multiplier=hidden_multiplier,
                    dist=dist,
                    save_coords=mmap_features)
                coords.append(coords_task)

                eval_task = client.compute(dask.delayed(eval_func, pure=False)(
                    model=cl_model,
                    graph=data,
                    embedding=coords_task.coordinates,
                    results_file=baseline_eval_file,
                    dist=dist,
                    device=device,
                    train_args=cl_train_args,
                    model_args=cl_model_args,
                    runs=cl_runs,
                    random_split=random_split,
                    mmap_features=mmap_features,
                    use_tmp=use_tmp), resources=gpu_req)
                eval_task.add_done_callback(progress_callback(baseline_progress))
                all_tasks.add(eval_task)
                del eval_task
                del coords_task


    patch_folder = output_folder / patch_folder_name(name, min_overlap, target_overlap, cluster, num_clusters,
                                                     num_iters, beta, levels, sparsify, target_patch_degree,
                                                     gamma)
    cluster_file = output_folder / cluster_file_name(name, cluster, num_clusters, num_iters, beta, levels)
    result_folder = patch_folder / result_folder_name
    result_folder.mkdir(exist_ok=True, parents=True)

    with SoftFileLock(patch_folder.with_suffix('.lock')):
        pg_exists = (patch_folder / 'patch_graph.pt').is_file()

    if not pg_exists:
        _load_data()
        patch_graph = dask.delayed(func.prepare_patches, pure=False)(
            output_folder=output_folder, name=name, graph=data,
            min_overlap=min_overlap, target_overlap=target_overlap,
            cluster=cluster,
            num_clusters=num_clusters, num_iters=num_iters, beta=beta, levels=levels,
            sparsify=sparsify, target_patch_degree=target_patch_degree,
            gamma=gamma,
            verbose=False).persist()
        patch_graph_initialised = True
        num_patches = patch_graph.num_nodes.compute()
    else:
        num_patches = torch.load(patch_folder / 'patch_graph.pt').num_nodes
        patch_graph = dask.delayed(torch.load, pure=True)(patch_folder / 'patch_graph.pt')
        patch_graph_initialised = False

    l2g_eval_file = result_folder / f'{eval_basename}_{l2g_name}_eval.json'
    n_nodes = None
    patch_data = None
    for d in dims:
        with ResultsDict(l2g_eval_file, lock=False) as res:
            r_done = res.runs(d)
        patch_runs = [[] for _ in range(num_patches)]
        for r in range(r_done, runs):
            _load_data()
            if not patch_graph_initialised:
                patch_graph = patch_graph.persist()

            if patch_data is None:
                patch_data = load_patch_data(patch_folder, data, num_patches)

            patches = []
            for pi in range(num_patches):
                patch_node_file = patch_folder / f'patch{pi}_index.npy'
                patch_result_file = result_folder / f'{train_basename}_patch{pi}_d{d}_r{r}_info.json'

                patch = dask.delayed(func.train, nout=2)(data=patch_data[pi], model=model,
                                                  lr=lr[r], num_epochs=num_epochs,
                                                  patience=patience, verbose=verbose_train,
                                                  results_file=patch_result_file, dim=d,
                                                  hidden_multiplier=hidden_multiplier, dist=dist,
                                                  save_coords=mmap_features)
                patches.append(patch)
                del patch

            l2g_coords_file = result_folder / f'{train_basename}_d{d}_r{r}_{l2g_name}_coords.npy'
            if l2g_coords_file.is_file():
                if mmap_features:
                    l2g_coords = l2g_coords_file
                else:
                    l2g_coords = dask.delayed(np.load)(file=l2g_coords_file)
            else:
                if n_nodes is None:
                    n_nodes = data.num_nodes.compute()
                shape = (n_nodes, d)

                l2g_coords = func.hierarchical_l2g_align_patches(
                                         patch_graph=patch_graph,
                                         shape=shape,
                                         scale=scale,
                                         rotate=rotate,
                                         translate=translate,
                                         patches=patches,
                                         mmap=mmap_features,
                                         use_tmp=use_tmp,
                                         verbose=verbose_l2g,
                                         output_file=l2g_coords_file,
                                         cluster_file=cluster_file,
                                         resparsify=resparsify
                                        )

            coords_task = dask.delayed(eval_func, pure=False)(
                model=cl_model,
                graph=data,
                embedding=l2g_coords,
                results_file=l2g_eval_file,
                dist=dist,
                device=device,
                train_args=cl_train_args,
                model_args=cl_model_args,
                runs=cl_runs,
                random_split=random_split,
                mmap_features=mmap_features,
                use_tmp=use_tmp
            )
            coords_task = client.compute(coords_task, resources=gpu_req)
            coords_task.add_done_callback(progress_callback(eval_progress))
            all_tasks.add(coords_task)
            del coords_task
            del l2g_coords



    baseline_progress.refresh()
    patch_progress.refresh()
    align_progress.refresh()
    eval_progress.refresh()
    total_progress.refresh()

    # make sure to wait for all tasks to complete and report overall progress
    watch_progress(all_tasks)
    manager.stop()


if __name__ == '__main__':
    print('launching main training script')
    # run main script
    ScriptParser(run, True).run()
