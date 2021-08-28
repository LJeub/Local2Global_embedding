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
from typing import Optional
from local2global_embedding.embedding import reconstruction_auc
from local2global_embedding.classfication import Logistic, train, accuracy, TrainingData
from local2global_embedding.run.utils import ResultsDict, ScriptParser


def evaluate(data_file: str, embedding_file: str, results_file: str, dist=False, device: Optional[str]=None,
             num_epochs=10000, patience=20, lr=0.01, runs=50, batch_size=1000):
    print(f'evaluating {embedding_file}')
    data = torch.load(data_file)
    num_labels = data.y.max().item()+1
    coords = torch.as_tensor(torch.load(embedding_file), dtype=torch.float32)
    dim = coords.shape[1]
    auc = reconstruction_auc(coords, data, dist=dist)
    tdata = TrainingData(coords, data.y, num_unlabeled=0)
    acc = []
    for _ in range(runs):
        model = Logistic(dim, num_labels)
        model = train(tdata, model, num_epochs, batch_size, lr, early_stop_patience=patience, weight_decay=0.0,
                      device=device, alpha=0, beta=0)
        acc.append(accuracy(tdata, model))
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
    evaluate(*args, **kwargs)
