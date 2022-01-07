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

from local2global_embedding.run.scripts.utils import ScopedTemporaryFile
from local2global_embedding.classfication import MLP, train, validation_accuracy
from local2global_embedding.run.utils import load_classification_problem, get_or_init_client, SyncDict, ScriptParser
import numpy as np
from dask import delayed, compute

from local2global_embedding.run.once_per_worker import once_per_worker


def load_data(name, data_root, embedding_file, mmap_features=False, use_tmp=False, **kwargs):
    mode = 'r' if mmap_features else None
    x = np.load(embedding_file, mmap_mode=mode)
    prob = load_classification_problem(name, root=data_root, **kwargs)
    if use_tmp and mmap_features:
        tmp_file = ScopedTemporaryFile(prefix='coords_',
                                       suffix='.npy')  # path of temporary file that is automatically cleaned up when garbage-collected
        x_tmp = np.memmap(tmp_file, dtype=x.dtype, shape=x.shape)
        x_tmp[:] = x[:]
        x = x_tmp
        print('features moved to tmp storage')
        prob._tmp_file = tmp_file  # keep reference to tmp_file alive
    prob.x = x
    return prob


@delayed
def train_task(data, model_args, batch_size=100, **train_args):
    print(f'training MLP({model_args})')
    model = MLP(input_dim=data.num_features, output_dim=data.num_labels, **model_args)
    model = train(data, model, batch_size=batch_size, **train_args)
    acc = validation_accuracy(data, model, batch_size)
    print(f'MLP({model_args}) achieved {acc=}')
    return acc


def mlp_grid_search(name, data_root, embedding_file, results_file, train_args={}, mmap_features=False,
                    use_tmp=False, data_args={}, **kwargs):
    """
    Run grid search over MLP parameters

    Args:
        name: Name of data set
        data_root: Root folder for downloaded data
        embedding_file: File containing embedding coordinates (npy)
        results_file: File to store search results (json)
        train_args: dict of parameters to pass to training function for all searches
        mmap_features: if True use mmap to load features
        use_tmp: if True and using mmap, copy features to temporary storage
        **kwargs: optionally override grid of parameters
        (default: kwargs = {'hidden_dim': (128, 256, 512, 1024), 'n_layers': (2, 3, 4), 'dropout': (0, 0.25, 0.5)})

    Returns: dictionary of best model parameters

    """
    client = get_or_init_client()  # launch distributed scheduler if run standalone
    results_file = Path(results_file)
    if results_file.is_file():
        with SyncDict(results_file, lock=False) as results:
            acc_list = results['acc_val']
            arg_list = results['model_args']
    else:
        grid = {'hidden_dim': (128, 256, 512, 1024), 'n_layers': (2, 3, 4), 'dropout': (0, 0.25, 0.5)}
        grid.update(kwargs)
        prob = once_per_worker(lambda: load_data(name, data_root, embedding_file, mmap_features, use_tmp, **data_args))
        acc_list = []
        arg_list = []

        for vals in product(*grid.values()):
            args = dict(zip(grid.keys(), vals))
            acc_list.append(train_task(prob, args, **train_args))
            arg_list.append(args)
        acc_list = compute(acc_list)
        with SyncDict(results_file, lock=True) as results:
            results['acc_val'] = acc_list
            results['model_args'] = arg_list
    acc, args = max(zip(acc_list, arg_list), key=lambda x: x[0])
    print(f'best model is MLP({args}) with {acc=}')
    return args


if __name__ == '__main__':
    ScriptParser(mlp_grid_search).run()
