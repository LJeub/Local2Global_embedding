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
from statistics import mean, stdev

import torch
import numpy as np
from numpy.lib.format import open_memmap
from typing import Optional

from local2global_embedding.embedding.eval import reconstruction_auc
from local2global_embedding.classfication import Logistic, train, accuracy, MLP
from local2global_embedding.run.utils import ResultsDict, ScriptParser, load_classification_problem, load_data
from .utils import ScopedTemporaryFile


def evaluate(name: str, data_root: str, restrict_lcc: bool, embedding_file: str, results_file: str, dist=False,
             device: Optional[str]=None, runs=50, train_args={},
             mmap_edges: Optional[str] = None, mmap_features: Optional[str] = None, random_split=False, use_tmp=False,
             model='logistic', model_args={}):
    train_args_default = dict(num_epochs=10000, patience=20, lr=0.01, batch_size=100000, alpha=0, beta=0, weight_decay=0)
    train_args_default.update(train_args)
    train_args = train_args_default

    print(f'evaluating {embedding_file} with {runs} classification runs.')
    graph = load_data(name, root=data_root, mmap_edges=mmap_edges, mmap_features=mmap_features,
                      restrict_lcc=restrict_lcc, load_features=False)
    cl_data = load_classification_problem(name, graph_args={'mmap_edges': mmap_edges, 'mmap_features': mmap_features},
                                          root=data_root, restrict_lcc=restrict_lcc)
    num_labels = cl_data.num_labels
    coords = np.load(embedding_file, mmap_mode=mmap_features)
    if use_tmp and mmap_features is not None:
        tmp_file = ScopedTemporaryFile(prefix='coords_', suffix='.npy')  # path of temporary file that is automatically cleaned up when garbage-collected
        coords_tmp = open_memmap(tmp_file, mode='w+', dtype=coords.dtype, shape=coords.shape)
        coords_tmp[:] = coords[:]
        coords = coords_tmp

    cl_data.x = torch.as_tensor(coords, dtype=torch.float32)
    dim = coords.shape[1]
    auc = reconstruction_auc(coords, graph, dist=dist)
    acc = []
    if model == 'logistic':
        def construct_model():
            return Logistic(dim, num_labels, **model_args)
    elif model == 'mlp':
        if 'hidden_dim' in model_args:
            def construct_model():
                return MLP(dim, output_dim=num_labels, **model_args)
        else:
            def construct_model():
                return MLP(dim, dim, num_labels, **model_args)
    else:
        raise ValueError(f'unknown model type {model}')

    for _ in range(runs):
        if random_split:
            cl_data.resplit()
        model = construct_model()
        model = train(cl_data, model, **train_args,
                      device=device)
        acc.append(accuracy(cl_data, model))
        print(f'Model accuracy: {acc[-1]}, max memory: {torch.cuda.max_memory_allocated()}, total available memory: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory}')
    acc_mean = mean(acc)
    if len(acc) == 1:
        acc_std = 0.
    else:
        acc_std = stdev(acc)
    with ResultsDict(results_file, replace=True) as results:
        results.update_dim(dim, auc=auc, acc_mean=acc_mean, acc_std=acc_std)


if __name__ == '__main__':
    parser = ScriptParser(evaluate)
    args, kwargs = parser.parse()
    evaluate(**kwargs)
