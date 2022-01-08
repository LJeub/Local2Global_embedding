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
from local2global_embedding.run.utils import get_or_init_client, SyncDict, ScriptParser
import numpy as np
from dask import delayed, compute

from local2global_embedding.run.once_per_worker import once_per_worker
from local2global_embedding.run.scripts.utils import load_cl_data


@delayed
def train_task(data, model_args, batch_size=100, **train_args):
    print(f'training MLP({model_args}) with parameters {train_args}')
    model = MLP(input_dim=data.num_features, output_dim=data.num_labels, **model_args)
    model = train(data, model, batch_size=batch_size, **train_args)
    acc = validation_accuracy(data, model, batch_size)
    print(f'MLP({model_args}) with parameters {train_args} achieved {acc=}')
    return acc


def mlp_grid_search(name, data_root, embedding_file, results_file, model_args=None, train_args=None,
                    mmap_features=False, use_tmp=False, data_args={}):
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
    client = get_or_init_client()  # launch distributed scheduler if run standalone
    results_file = Path(results_file)
    if results_file.is_file():
        with SyncDict(results_file, lock=False) as results:
            acc_list = results['acc_val']
            marg_list = results['model_args']
            targ_list = results['training_args']
    else:
        model_grid = {'hidden_dim': (128, 256, 512, 1024), 'n_layers': (2, 3, 4), 'dropout': (0, 0.25, 0.5), 'batch_norm': (True,)}
        if model_args is not None:
            model_grid.update(model_args)
        train_grid = {'batch_size': (100000,), 'num_epochs': (1000,), 'patience': (20,), 'lr': (0.01, 0.001, 0.0001)}
        if train_args is not None:
            train_grid.update(train_args)
        prob = once_per_worker(lambda: load_cl_data(name, data_root, embedding_file, mmap_features, use_tmp, **data_args))
        acc_list = []
        marg_list = []
        targ_list = []
        for tvals in product(*train_grid.values()):
            targs = dict(zip(train_grid.keys(), tvals))
            for vals in product(*model_grid.values()):
                args = dict(zip(model_grid.keys(), vals))
                acc_list.append(train_task(prob, args, **targs))
                marg_list.append(args)
                targ_list.append(targs)
        acc_list = compute(*acc_list)
        with SyncDict(results_file, lock=True) as results:
            results['acc_val'] = acc_list
            results['model_args'] = marg_list
            results['train_args'] = targ_list
    acc, margs, targs = max(zip(acc_list, marg_list, targ_list), key=lambda x: x[0])
    print(f'best model is MLP({margs}) trained with {targs}, {acc=}')
    return margs, targs


if __name__ == '__main__':
    ScriptParser(mlp_grid_search).run()
