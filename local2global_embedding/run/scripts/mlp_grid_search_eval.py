#  Copyright (c) 2022. Lucas G. S. Jeub
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
import torch
from pathlib import Path
from itertools import product
from collections.abc import Iterable

from local2global_embedding.embedding.eval import reconstruction_auc
from local2global_embedding.classfication import MLP, train, accuracy
from local2global_embedding.run.utils import get_or_init_client, ScriptParser, ResultsDict, load_data
import numpy as np
from dask import delayed, compute

from local2global_embedding.run.once_per_worker import once_per_worker
from local2global_embedding.run.scripts.utils import load_cl_data


@delayed
def train_task(data, model_args, results_file, batch_size=100, device=None, **train_args):
    results_file = Path(results_file)
    print(f'training MLP({model_args}) with parameters {train_args}')
    model = MLP(input_dim=data.num_features, output_dim=data.num_labels, **model_args)
    model = train(data, model, batch_size=batch_size, device=None, **train_args)
    val_acc = accuracy(data, model, batch_size=batch_size, mode='val')
    train_args['batch_size'] = batch_size
    with ResultsDict(results_file, lock=True) as results:
        if val_acc > results.max('val_acc', dim=data.num_features):
            torch.save(model, results_file.with_name(results_file.stem + f'_d{data.num_features}_bestclassifier.pt'))
        results.update_dim(data.num_features, val_acc=val_acc, model_args=model_args, train_args=train_args)
    print(f'MLP({model_args}) with parameters {train_args} achieved {val_acc=}')
    return val_acc


@delayed
def compute_auc(name, data_root, restrict_lcc, mmap_edges, coords, dist):
    graph = load_data(name, data_root, restrict_lcc=restrict_lcc, mmap_edges=mmap_edges, load_features=False)
    return reconstruction_auc(coords, graph, dist)


def _clean_grid_args(args):
    grid = {}
    for key, val in args.items():
        if isinstance(val, Iterable):
            grid[key] = val
        else:
            grid[key] = (val,)
    return grid

def _make_grid(model_args, train_args):
    model_args = _clean_grid_args(model_args)
    train_args = _clean_grid_args(train_args)
    return [(dict(zip(model_args.keys(), vals)), dict(zip(train_args.keys(), tvals)))
            for tvals in product(*train_args.values())
            for vals in product(*model_args.values())]


def mlp_grid_search_eval(name, data_root, embedding_file, results_file, dist=False, model_args=None, train_args=None,
                         mmap_edges=None, mmap_features=None, random_split=False, use_tmp=False, model='mlp',
                         restrict_lcc=False, device=None, runs=None):
    """
    Run grid search over MLP parameters

    Args:
        name: Name of data set
        data_root: Root folder for downloaded data
        embedding_file: File containing embedding coordinates (npy)
        results_file: File to store search results (json)
        train_args: grid of training arguments default ({'batch_size': (100000,), 'num_epochs': (1000,), 'patience': (20,), 'lr': (0.01, 0.001, 0.0001)})
        mmap_features: if True use mmap to load features
        use_tmp: if True and using mmap, copy features to temporary storage
        model_args: grid of model parameters
        (default: kwargs = {'hidden_dim': (128, 256, 512, 1024), 'n_layers': (2, 3, 4), 'dropout': (0, 0.25, 0.5),
                            'batch_norm': (True,)})

    Returns: dictionary of best model parameters

    """
    # TODO implement multiple runs with random split
    results_file = Path(results_file)
    final_results_file = results_file.with_name(results_file.stem + '_best.json')
    dim = np.load(embedding_file, mmap_mode='r').shape[1]
    if model != 'mlp':
        raise NotImplementedError('grid search only implemented for MLP')
    with ResultsDict(final_results_file, lock=True) as best_results:
        if best_results.contains_dim(dim):
            return

    client = get_or_init_client()  # launch distributed scheduler if run standalone
    model_grid = {'hidden_dim': (128, 256, 512, 1024), 'n_layers': (2, 3, 4), 'dropout': (0, 0.25, 0.5),
                  'batch_norm': (True,)}
    if model_args is not None:
        model_grid.update(model_args)
    train_grid = {'batch_size': (100000,), 'num_epochs': (1000,), 'patience': (20,), 'lr': (0.01, 0.001, 0.0001)}
    if train_args is not None:
        train_grid.update(train_args)
    arg_grid = _make_grid(model_grid, train_grid)
    if results_file.is_file():
        with ResultsDict(results_file, lock=False) as results:
            runs = results.runs(dim)
            arg_grid = arg_grid[runs:]  # resume remaining experiments
    prob = once_per_worker(lambda: load_cl_data(name, data_root, embedding_file, mmap_features, use_tmp,
                                                restrict_lcc=restrict_lcc))
    task_list = []
    for margs, targs in arg_grid:
            task_list.append(train_task(prob, margs, results_file, **targs))
    auc = compute_auc(name, data_root, restrict_lcc, mmap_edges, prob.x, dist)
    task_list, auc = compute(task_list, auc)

    with ResultsDict(results_file) as results:
        with ResultsDict(final_results_file, lock=True, replace=True) as best_results:
            best_model = torch.load(results_file.with_name(results_file.stem + f'_d{dim}_bestclassifier.pt'))
            test_acc = accuracy(prob.compute(), best_model, mode='test')
            val_list = results.get('val_acc', dim=dim)
            i = np.argmax(val_list)
            best_model_args = results.get('model_args', dim=dim)[i]
            best_train_args = results.get('train_args', dim=dim)[i]
            best_results.update_dim(dim, test_acc=test_acc, best_model_args=best_model_args,
                               best_train_args=best_train_args)
            print(f'best model is MLP({best_model_args} trained with {best_train_args}, {test_acc=}')


if __name__ == '__main__':
    ScriptParser(mlp_grid_search_eval).run()
