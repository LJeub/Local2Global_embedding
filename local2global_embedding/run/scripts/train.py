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
from typing import Optional
import os
from collections.abc import Iterable
from time import perf_counter

import numpy as np
import torch

import local2global_embedding.embedding.gae as gae
import local2global_embedding.embedding.dgi as dgi
import local2global_embedding.embedding.train as training
from local2global_embedding.embedding.eval import reconstruction_auc
from local2global_embedding.utils import speye, set_device
from local2global_embedding.run.utils import ResultsDict, ScriptParser


def select_loss(model):
    if isinstance(model, gae.VGAE):
        return gae.VGAE_loss
    elif isinstance(model, gae.GAE):
        return gae.GAE_loss
    elif isinstance(model, dgi.DGI):
        return dgi.DGILoss()


def create_model(model, dim, hidden_dim, num_features, dist):
    if model == 'VGAE':
        return gae.VGAE(dim, hidden_dim, num_features, dist)
    elif model == 'GAE':
        return gae.GAE(dim, hidden_dim, num_features, dist)
    elif model == 'DGI':
        return dgi.DGI(num_features, dim)


class Count:
    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1


def train(data, model, lr, num_epochs: int, patience: int, verbose: bool, results_file: str,
          dim: int, hidden_multiplier: Optional[int] = None, no_features=False, dist=False,
          device: Optional[str] = None, runs=1, normalise_features=False):
    """
    train model on data

    Args:
        data: path to training data
        model: str that will be evaluated to initialise the model
        lr: learning rate
        num_epochs: maximum number of training epochs
        patience: early stopping patience
        verbose: if True, print loss during training
        results_file: json file of existing results
        dist: use distance decoder for reconstruction
        device: device to use for training (e.g., 'cuda', 'cpu')
    """
    device = set_device(device)
    data_str = data
    model_str = model
    if not isinstance(lr, Iterable):
        lr = [lr] * runs

    print(f'Launched training for {data} and model {model}_d{dim} with cuda devices {os.environ.get("CUDA_VISIBLE_DEVICES", "unavailiable")} and device={device}')
    data = torch.load(data).to(device)
    results_file = Path(results_file)

    if no_features:
        data.x = speye(data.num_nodes).to(device)
    else:
        data.x = data.x.to(torch.float32)
        if normalise_features:
            r_sum = data.x.sum(dim=1)
            r_sum[r_sum == 0] = 1.0  # avoid division by zero
            data.x /= r_sum[:, None]

    model_auc_file = results_file.with_name(results_file.name.replace('_info.json', f'_d{dim}_best_auc_model.pt'))
    model_loss_file = model_auc_file.with_name(model_auc_file.name.replace('_auc_', '_loss_'))
    coords_auc_file = model_auc_file.with_name(model_auc_file.name.replace('model.pt', 'coords.npy'))
    coords_loss_file = model_auc_file.with_name(model_loss_file.name.replace('model.pt', 'coords.npy'))
    model = create_model(model, dim, dim * hidden_multiplier, data.num_features, dist).to(device)
    loss_fun = select_loss(model)

    with ResultsDict(results_file) as results:
        runs_done = results.runs(dim)

    updated_auc = False
    updated_loss = False
    for r in range(runs_done, runs):
        tic = perf_counter()
        model.reset_parameters()
        ep_count = Count()
        model = training.train(data, model, loss_fun, num_epochs, patience, lr[r], verbose=verbose, logger=ep_count)
        coords = model.embed(data)
        toc = perf_counter()

        auc = reconstruction_auc(coords, data, dist=dist)
        loss = float(loss_fun(model, data))

        with ResultsDict(results_file) as results:
            print(f'Training for run {results.runs(dim)+1} of {data_str} and model {model_str}_d{dim} stopped after {ep_count.count} epochs')
            if results.runs(dim) >= runs:
                break
            if results.min('loss', dim) > loss:
                updated_loss = True
                torch.save(model.state_dict(), model_loss_file)
                np.save(coords_loss_file, coords.cpu().numpy())
            if results.max('auc', dim) < auc:
                updated_auc = True
                torch.save(model.state_dict(), model_auc_file)
                np.save(coords_auc_file, coords.cpu().numpy())
            results.update_dim(dim, auc=auc, loss=loss, args={'lr': lr[r], 'num_epochs': num_epochs,
                                                              'patience': patience, 'dist': dist, 'time': toc-tic})
            runs_done = results.runs(dim)

    return coords_auc_file, updated_auc, coords_loss_file, updated_loss


if __name__ == '__main__':
    ScriptParser(train).run()
