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

import torch

from local2global_embedding.run.utils import ResultsDict, load_data, ScriptParser, run_script, patch_folder_name


async def run(name='Cora', data_root='/tmp', no_features=False, model='VGAE', num_epochs=10000,
              patience=20, runs=10, dims: List[int] = None, hidden_multiplier=2, target_patch_degree=4.0,
              min_overlap: int = None, target_overlap: int = None, gamma=0.0, sparsify='resistance',
              cluster='metis', num_clusters=10, beta=0.1, num_iters: int = None, lr=0.01, dist=False,
              output='.', device: str = None, plot=False, verbose=False, max_workers=1, cmd_prefix: str = None,
              run_baseline=True):
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

    if dims is None:
        dims = [2]
    output_folder = Path(output)
    data_file = output_folder / f'{name}_data.pt'
    if not data_file.is_file():
        data = load_data(name, data_root)
        torch.save(data, data_file)

    basename = f'{name}_{model}'
    min_overlap = min_overlap if min_overlap is not None else max(dims) + 1
    target_overlap = target_overlap if target_overlap is not None else 2 * max(dims)

    if dist:
        basename += '_dist'

    patch_create_task = asyncio.create_task(run_script('prepare_patches', cmd_prefix=cmd_prefix, task_queue=work_queue,
                                                       output_folder=output_folder, name=name,
                                                       min_overlap=min_overlap, target_overlap=target_overlap,
                                                       cluster=cluster,
                                                       num_clusters=num_clusters, num_iters=num_iters, beta=beta,
                                                       sparsify=sparsify, target_patch_degree=target_patch_degree,
                                                       gamma=gamma,
                                                       verbose=False))

    # compute baseline full model if necessary
    baseline_info_file = output_folder / f'{basename}_full_info.json'
    baseline_eval_file = output_folder / f'{basename}_full_eval.json'
    baseline_coords_to_evaluate = set()
    baseline_tasks = []
    if run_baseline:
        for d in dims:
            with ResultsDict(baseline_info_file) as baseline_data:
                r = baseline_data.runs(d)
            if not baseline_eval_file.is_file() or r < runs:
                baseline_coords_to_evaluate.add(output_folder / f'{basename}_full_d{d}_best_coords.pt')
            if r < runs:
                print(f'training full model for {runs - r} runs and d={d}')
                for r_it in range(r, runs):
                    baseline_tasks.append(
                        asyncio.create_task(run_script('train', cmd_prefix=cmd_prefix, task_queue=work_queue,
                                                       data=data_file, model=model,
                                                       lr=lr, num_epochs=num_epochs,
                                                       patience=patience, verbose=verbose,
                                                       results_file=baseline_info_file, dim=d,
                                                       hidden_multiplier=hidden_multiplier,
                                                       no_features=no_features, dist=dist,
                                                       device=device)))

    patch_folder = output_folder / patch_folder_name(name, min_overlap, target_overlap, cluster, num_clusters,
                                                     num_iters, beta, sparsify, target_patch_degree,
                                                     gamma)

    # Compute patch embeddings
    await asyncio.gather(patch_create_task)  # make sure patch data is available
    patch_tasks = []
    l2g_eval_file = patch_folder / f'{basename}_l2g_eval.json'
    nt_eval_file = patch_folder / f'{basename}_nt_eval.json'
    l2g_coords_to_evaluate = set()
    nt_coords_to_evaluate = set()
    compute_alignment_for_dims = set()
    for d in dims:
        for patch_data_file in patch_folder.glob('patch*_data.pt'):
            patch_id = patch_data_file.stem.replace('_data', '')
            patch_result_file = patch_folder / f'{basename}_{patch_id}_info.json'
            with ResultsDict(patch_result_file) as patch_results:
                r = patch_results.runs(d)
            if not l2g_eval_file.is_file():
                l2g_coords_to_evaluate.add(patch_folder / f'{basename}_d{d}_coords.pt')
            if not nt_eval_file.is_file():
                nt_coords_to_evaluate.add(patch_folder / f'{basename}_d{d}_ntcoords.pt')
            if r < runs:
                compute_alignment_for_dims.add(d)
                l2g_coords_to_evaluate.add(patch_folder / f'{basename}_d{d}_coords.pt')
                nt_coords_to_evaluate.add(patch_folder / f'{basename}_d{d}_ntcoords.pt')
                print(f'training {patch_id} for {runs - r} runs and d={d}')
                for r_it in range(r, runs):
                    patch_tasks.append(
                        asyncio.create_task(
                            run_script('train', cmd_prefix=cmd_prefix, task_queue=work_queue,
                                       data=patch_data_file, model=model,
                                       lr=lr, num_epochs=num_epochs,
                                       patience=patience, verbose=verbose,
                                       results_file=patch_result_file, dim=d,
                                       hidden_multiplier=hidden_multiplier,
                                       no_features=no_features, dist=dist,
                                       device=device)))

    # local2global alignment of patch embeddings
    await asyncio.gather(*patch_tasks)
    alignment_tasks = []
    for d in compute_alignment_for_dims:
        alignment_tasks.append(
            asyncio.create_task(run_script('l2g_align_patches', cmd_prefix=cmd_prefix, task_queue=work_queue,
                                           patch_folder=patch_folder, basename=basename, dim=d)))

    # evaluate embeddings
    await asyncio.gather(*baseline_tasks)  # make sure baseline data is available
    eval_tasks = []
    for coords_file in baseline_coords_to_evaluate:
        eval_tasks.append(
            asyncio.create_task(
                run_script('evaluate', cmd_prefix=cmd_prefix, task_queue=work_queue,
                           data_file=data_file,
                           embedding_file=coords_file,
                           results_file=baseline_eval_file,
                           dist=dist,
                           device=device,
                           )
            )
        )

    await asyncio.gather(*alignment_tasks)  # make sure aligned coordinates are available
    for coords_file in l2g_coords_to_evaluate:
        eval_tasks.append(
            asyncio.create_task(
                run_script('evaluate', cmd_prefix=cmd_prefix, task_queue=work_queue,
                           data_file=data_file,
                           embedding_file=coords_file,
                           results_file=l2g_eval_file,
                           dist=dist,
                           device=device,
                           )
            )
        )

    for coords_file in nt_coords_to_evaluate:
        eval_tasks.append(
            asyncio.create_task(
                run_script('evaluate', cmd_prefix=cmd_prefix, task_queue=work_queue,
                           data_file=data_file,
                           embedding_file=coords_file,
                           results_file=nt_eval_file,
                           dist=dist,
                           device=device,
                           )
            )
        )
    await asyncio.gather(*eval_tasks)


if __name__ == '__main__':
    # run main script
    parser = ScriptParser(run)
    args, kwargs = parser.parse()
    asyncio.run(run(*args, **kwargs))
