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
from pathlib import Path

from local2global_embedding.classfication import hyper_tune
from local2global_embedding.run.utils import load_classification_problem, ScriptParser
from local2global_embedding.run.scripts.utils import load_cl_data
import numpy as np


def hyper_tune_parameters(name, data_root, embedding_file, results_file, runs=100, train_params=None, model_params=None,
                          mmap_features=None,
                          use_tmp=False, **kwargs):
    results_file = Path(results_file)

    prob = load_cl_data(name, data_root, embedding_file, mmap_features, use_tmp, **kwargs)
    m_args, t_args, results = hyper_tune(prob, max_evals=runs, train_args=train_params, model_args=model_params)
    results.to_csv(results_file)
    arg_index = results['loss'].argmax()



if __name__ == '__main__':
    ScriptParser(hyper_tune_parameters).run()
