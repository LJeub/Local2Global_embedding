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
from local2global_embedding.classfication import hyper_tune
from local2global_embedding.run.utils import load_classification_problem, ScriptParser
from local2global_embedding.run.scripts.utils import ScopedTemporaryFile
import numpy as np


def hyper_tune_parameters(name, data_root, embedding_file, results_file, runs=100, train_params={}, mmap_features=False,
                          use_tmp=False):
    mode = 'r' if mmap_features else None
    x = np.load(embedding_file, mmap_mode=mode)
    if use_tmp and mmap_features:
        tmp_file = ScopedTemporaryFile(prefix='coords_', suffix='.npy')  # path of temporary file that is automatically cleaned up when garbage-collected
        x_tmp = np.memmap(tmp_file, dtype=x.dtype, shape=x.shape)
        x_tmp[:] = x[:]
        x = x_tmp
        print('features moved to tmp storage')
    prob = load_classification_problem(name, root=data_root)
    prob.x = x
    results, obj = hyper_tune(prob, max_evals=runs, **train_params)
    results.to_csv(results_file)


if __name__ == '__main__':
    ScriptParser(hyper_tune_parameters).run()