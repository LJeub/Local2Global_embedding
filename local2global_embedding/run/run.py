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

from pathlib import Path
import asyncio
from typing import List
import sys

import torch
from tqdm.asyncio import tqdm_asyncio

from local2global_embedding.run.utils import ResultsDict, load_data, ScriptParser, run_script, patch_folder_name, Throttler


async def run(name='Cora', data_root='/tmp', no_features=False, model='VGAE', num_epochs=10000,
              patience=20, runs=10, cl_runs=50, dims: List[int] = None, hidden_multiplier=2, target_patch_degree=4.0,
              min_overlap: int = None, target_overlap: int = None, gamma=0.0, sparsify='resistance',
              cluster='metis', num_clusters=10, beta=0.1, num_iters: int = None, lr=0.001, cl_lr=0.01, dist=False,
              output='.', device: str = None, verbose=False, max_workers=1, cmd_prefix: str = None,
              run_baseline=True, normalise=False, restrict_lcc=False, mmap_edges=False, mmap_features=False,
              random_split=False, use_tmp=False, max_frequency=0.):
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

    work_queue = asyncio.Queue(maxsize=max_workers)  # used to control the maximum number of simultaneous processes
    throttler = Throttler(max_frequency)

    if dims is None:
        dims = [2]
    output_folder = Path(output).expanduser()
    data_root = Path(data_root).expanduser()
    print(f'Started experiment for data set {name}.')
    print(f'Results will be placed in {output_folder}.')
    print(f'Data root is {data_root}')
    if use_tmp:
        print('Any memmapped data will be moved to local storage.')

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

    patch_create_task = asyncio.create_task(run_script('prepare_patches', _cmd_prefix=cmd_prefix,
                                                       _task_queue=work_queue, _throttler=throttler,
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
                                                       mmap_features=mmap_features))

    # compute baseline full model if necessary
    baseline_info_file = output_folder / f'{train_basename}_full_info.json'
    baseline_dims_to_evaluate = set()
    baseline_tasks = []
    if run_baseline:
        data_file = output_folder / f'{name}_data.pt'
        if not data_file.is_file():
            data = load_data(name, data_root, restrict_lcc=restrict_lcc, normalise=normalise, mmap_edges=mmap_edges,
                             mmap_features=mmap_features)
            torch.save(data, data_file)
        for d in dims:
            with ResultsDict(baseline_info_file) as baseline_data:
                r = baseline_data.runs(d)

            if r < runs:
                baseline_dims_to_evaluate.add(d)
                baseline_tasks.append(
                    asyncio.create_task(run_script('train', _cmd_prefix=cmd_prefix, _task_queue=work_queue,
                                                   _throttler=throttler, _stderr=True,
                                                   data=data_file, model=model,
                                                   lr=lr, num_epochs=num_epochs,
                                                   patience=patience, verbose=verbose,
                                                   results_file=baseline_info_file, dim=d,
                                                   hidden_multiplier=hidden_multiplier,
                                                   no_features=no_features, dist=dist,
                                                   device=device, runs=runs)))

    patch_folder = output_folder / patch_folder_name(name, min_overlap, target_overlap, cluster, num_clusters,
                                                     num_iters, beta, sparsify, target_patch_degree,
                                                     gamma)

    # Compute patch embeddings
    await asyncio.gather(patch_create_task)  # make sure patch data is available
    patch_tasks = []

    compute_alignment_for_dims = set()
    for d in dims:
        for patch_data_file in patch_folder.glob('patch*_data.pt'):
            patch_id = patch_data_file.stem.replace('_data', '')
            patch_result_file = patch_folder / f'{train_basename}_{patch_id}_info.json'
            with ResultsDict(patch_result_file) as patch_results:
                r = patch_results.runs(d)
            if r < runs:
                compute_alignment_for_dims.add(d)
                patch_tasks.append(
                    asyncio.create_task(
                        run_script('train', _cmd_prefix=cmd_prefix, _task_queue=work_queue, _throttler=throttler,
                                   _stderr=True,
                                   data=patch_data_file, model=model,
                                   lr=lr, num_epochs=num_epochs,
                                   patience=patience, verbose=verbose,
                                   results_file=patch_result_file, dim=d,
                                   hidden_multiplier=hidden_multiplier,
                                   no_features=no_features, dist=dist,
                                   device=device, runs=runs)))

    # local2global alignment of patch embeddings
    print('running patch tasks')
    await tqdm_asyncio.gather(*patch_tasks, total=len(patch_tasks), file=sys.stdout, smoothing=0)
    alignment_tasks = []
    l2g_dims_to_evaluate = set()
    for d in dims:
        for criterion in ('auc', 'loss'):
            coords_files = [patch_folder / f'{train_basename}_d{d}_{criterion}_{nt}coords.npy'
                            for nt in ('', 'nt')]
            if d in compute_alignment_for_dims or not all(coords_file.is_file() for coords_file in coords_files):
                l2g_dims_to_evaluate.add(d)
                alignment_tasks.append(
                    asyncio.create_task(run_script('l2g_align_patches', _cmd_prefix=cmd_prefix, _task_queue=work_queue,
                                                   _throttler=throttler, _stderr=True,
                                                   patch_folder=patch_folder, basename=train_basename, dim=d,
                                                   criterion=criterion,
                                                   mmap=mmap_features is not None, use_tmp=use_tmp)))

    # evaluate embeddings
    print('running baseline tasks')
    await tqdm_asyncio.gather(*baseline_tasks, total=len(baseline_tasks), file=sys.stdout, smoothing=0)  # make sure baseline data is available
    eval_tasks = []
    if run_baseline:
        baseline_loss_eval_file = output_folder / f'{eval_basename}_full_loss_eval.json'
        baseline_auc_eval_file = baseline_loss_eval_file.with_name(baseline_loss_eval_file.name.replace('_loss_','_auc_'))
        for d in dims:
            with ResultsDict(baseline_loss_eval_file, replace=True) as eval_results:
                if d in baseline_dims_to_evaluate or not eval_results.contains_dim(d):
                    coords_file = output_folder / f'{train_basename}_full_d{d}_best_loss_coords.npy'
                    eval_tasks.append(
                        asyncio.create_task(
                            run_script('evaluate', _cmd_prefix=cmd_prefix, _task_queue=work_queue,
                                       _throttler=throttler, _stderr=True,
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
                                       mmap_features=mmap_features,

                                       )
                        )
                    )
            with ResultsDict(baseline_auc_eval_file, replace=True) as eval_results:
                if d in baseline_dims_to_evaluate or not eval_results.contains_dim(d):
                    coords_file = output_folder / f'{train_basename}_full_d{d}_best_auc_coords.npy'
                    eval_tasks.append(
                        asyncio.create_task(
                            run_script('evaluate', _cmd_prefix=cmd_prefix, _task_queue=work_queue,
                                       _throttler=throttler, _stderr=True,
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
                        )
                    )

    print('running alignment tasks')
    await tqdm_asyncio.gather(*alignment_tasks, total=len(alignment_tasks),  file=sys.stdout, smoothing=0)  # make sure aligned coordinates are available
    l2g_loss_eval_file = patch_folder / f'{eval_basename}_l2g_loss_eval.json'
    l2g_auc_eval_file = l2g_loss_eval_file.with_name(l2g_loss_eval_file.name.replace('_loss_', '_auc_'))
    nt_loss_eval_file = patch_folder / f'{eval_basename}_nt_loss_eval.json'
    nt_auc_eval_file = nt_loss_eval_file.with_name(nt_loss_eval_file.name.replace('_loss_', '_auc_'))
    for d in dims:
        with ResultsDict(l2g_loss_eval_file, replace=True) as l2g_eval:
            if d in l2g_dims_to_evaluate or not l2g_eval.contains_dim(d):
                coords_file = patch_folder / f'{train_basename}_d{d}_loss_coords.npy'
                eval_tasks.append(
                    asyncio.create_task(
                        run_script('evaluate', _cmd_prefix=cmd_prefix, _task_queue=work_queue,
                                   _throttler=throttler, _stderr=True,
                                   name=name,
                                   data_root=data_root,
                                   restrict_lcc=restrict_lcc,
                                   embedding_file=coords_file,
                                   results_file=l2g_loss_eval_file,
                                   dist=dist,
                                   device=device,
                                   lr=cl_lr,
                                   runs=cl_runs,
                                   random_split=random_split,
                                   mmap_edges=mmap_edges,
                                   mmap_features=mmap_features,
                                   )
                    )
                )

        with ResultsDict(l2g_auc_eval_file, replace=True) as l2g_eval:
            if d in l2g_dims_to_evaluate or not l2g_eval.contains_dim(d):
                coords_file = patch_folder / f'{train_basename}_d{d}_auc_coords.npy'
                eval_tasks.append(
                    asyncio.create_task(
                        run_script('evaluate', _cmd_prefix=cmd_prefix, _task_queue=work_queue,
                                   _throttler=throttler, _stderr=True,
                                   name=name,
                                   data_root=data_root,
                                   restrict_lcc=restrict_lcc,
                                   embedding_file=coords_file,
                                   results_file=l2g_auc_eval_file,
                                   dist=dist,
                                   device=device,
                                   lr=cl_lr,
                                   runs=cl_runs,
                                   random_split=random_split,
                                   mmap_edges=mmap_edges,
                                   mmap_features=mmap_features,
                                   )
                    )
                )

    for d in dims:
        with ResultsDict(nt_loss_eval_file, replace=True) as nt_eval:
            if d in l2g_dims_to_evaluate or not nt_eval.contains_dim(d):
                coords_file = patch_folder / f'{train_basename}_d{d}_loss_ntcoords.npy'
                eval_tasks.append(
                    asyncio.create_task(
                        run_script('evaluate', _cmd_prefix=cmd_prefix, _task_queue=work_queue,
                                   _throttler=throttler, _stderr=False,
                                   name=name,
                                   data_root=data_root,
                                   restrict_lcc=restrict_lcc,
                                   embedding_file=coords_file,
                                   results_file=nt_loss_eval_file,
                                   dist=dist,
                                   device=device,
                                   lr=cl_lr,
                                   runs=cl_runs,
                                   random_split=random_split,
                                   mmap_edges=mmap_edges,
                                   mmap_features=mmap_features,
                                   )
                    )
                )

        with ResultsDict(nt_auc_eval_file, replace=True) as nt_eval:
            if d in l2g_dims_to_evaluate or not nt_eval.contains_dim(d):
                coords_file = patch_folder / f'{train_basename}_d{d}_auc_ntcoords.npy'
                eval_tasks.append(
                    asyncio.create_task(
                        run_script('evaluate', _cmd_prefix=cmd_prefix, _task_queue=work_queue,
                                   _throttler=throttler, _stderr=True,
                                   name=name,
                                   data_root=data_root,
                                   restrict_lcc=restrict_lcc,
                                   embedding_file=coords_file,
                                   results_file=nt_auc_eval_file,
                                   dist=dist,
                                   device=device,
                                   lr=cl_lr,
                                   runs=cl_runs,
                                   random_split=random_split,
                                   mmap_edges=mmap_edges,
                                   mmap_features=mmap_features,
                                   )
                    )
                )
    print('evaluating embeddings')
    await tqdm_asyncio.gather(*eval_tasks, total=len(eval_tasks),  file=sys.stdout, smoothing=0)


if __name__ == '__main__':
    # run main script
    parser = ScriptParser(run)
    args, kwargs = parser.parse()
    asyncio.run(run(*args, **kwargs))
