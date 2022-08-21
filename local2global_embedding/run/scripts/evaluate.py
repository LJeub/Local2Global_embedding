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
from statistics import mean, stdev

import torch
from copy import copy
import numpy as np
from numpy.lib.format import open_memmap
from typing import Optional

from local2global_embedding.embedding.eval import reconstruction_auc
from local2global_embedding.classfication import Logistic, train, accuracy, MLP
from local2global_embedding.run.utils import ResultsDict, ScriptParser, load_classification_problem, load_data
from .utils import ScopedTemporaryFile
from traceback import print_exc


def evaluate(graph, embedding, results_file: str, dist=False,
             device: Optional[str]=None, runs=50, train_args={}, mmap_features=False, random_split=False,
             model='logistic', model_args={}, use_tmp=False):
    try:
        train_args_default = dict(num_epochs=10000, patience=20, lr=0.01, batch_size=100000, alpha=0, beta=0, weight_decay=0)
        train_args_default.update(train_args)
        train_args = train_args_default

        mmap_mode = 'r' if mmap_features else None

        if isinstance(embedding, str) or isinstance(embedding, Path):
            coords = np.load(embedding, mmap_mode=mmap_mode)
        else:
            coords = embedding
        print(f'evaluating with {runs} classification runs.')
        print('graph data loaded')
        cl_data = copy(graph.cl_data)
        print('classification problem loaded')
        num_labels = cl_data.num_labels

        if use_tmp and mmap_features:
            tmp_file = ScopedTemporaryFile(prefix='coords_', suffix='.npy')  # path of temporary file that is automatically cleaned up when garbage-collected
            coords_tmp = open_memmap(tmp_file, mode='w+', dtype=coords.dtype, shape=coords.shape)
            coords_tmp[:] = coords[:]
            coords = coords_tmp
            print('features moved to tmp storage')

        cl_data.x = torch.as_tensor(coords, dtype=torch.float32)
        dim = coords.shape[1]
        auc = reconstruction_auc(coords, graph, dist=dist)

        acc = []
        model_str = model
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
            model = train(cl_data, model, device=device, **train_args)
            acc.append(accuracy(cl_data, model))
            if torch.cuda.is_available():
                print(f'Model accuracy: {acc[-1]}, max memory: {torch.cuda.max_memory_allocated()}, total available memory: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory}')

        with ResultsDict(results_file, replace=False, lock=True) as results:
            results.update_dim(dim, auc=auc, acc=acc, model=model_str, train_args=train_args,
                               model_args=model_args)
    except Exception:
        print_exc()
        raise

if __name__ == '__main__':
    parser = ScriptParser(evaluate)
    args, kwargs = parser.parse()
    evaluate(**kwargs)
