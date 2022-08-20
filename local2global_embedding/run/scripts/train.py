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
import json
from pathlib import Path
from typing import Optional
import os
from collections.abc import Iterable
from time import perf_counter

import numpy as np
import torch
from atomicwrites import atomic_write

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
          device: Optional[str] = None, runs=1, normalise_features=False, save_coords=False):
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

    print(f'Launched training for {data} and model {model}_d{dim} with cuda devices {os.environ.get("CUDA_VISIBLE_DEVICES", "unavailable")} and device={device}')
    data = torch.load(data).to(device)
    results_file = Path(results_file)
    model_file = results_file.with_name(results_file.stem.replace("_info", "_model") + ".pt")
    coords_file = results_file.with_name(results_file.stem.replace("_info", "_coords") + ".npy")

    if no_features:
        data.x = speye(data.num_nodes).to(device)
    else:
        data.x = data.x.to(torch.float32)
        if normalise_features:
            r_sum = data.x.sum(dim=1)
            r_sum[r_sum == 0] = 1.0  # avoid division by zero
            data.x /= r_sum[:, None]

    model = create_model(model, dim, dim * hidden_multiplier, data.num_features, dist).to(device)
    loss_fun = select_loss(model)
    if results_file.exists():
        if coords_file.exists():
            if save_coords:
                return str(coords_file)
            else:
                return torch.from_numpy(np.load(coords_file))
        else:
            model.load_state_dict(torch.load(model_file))
            model.eval()
            coords = model.embed(data)
            if save_coords:
                np.save(coords_file, coords.cpu().numpy())
                return str(coords_file)
            else:
                return coords
    else:
        tic = perf_counter()
        model.reset_parameters()
        ep_count = Count()
        model = training.train(data, model, loss_fun, num_epochs, patience, lr, verbose=verbose, logger=ep_count)
        model.eval()
        coords = model.embed(data)
        toc = perf_counter()
        auc = reconstruction_auc(coords, data, dist=dist)
        loss = float(loss_fun(model, data))
        torch.save(model.state_dict(), model_file)
        if save_coords:
            np.save(coords_file, coords.cpu().numpy())
        with atomic_write(results_file, overwrite=True) as f:  # this should avoid any chance of loosing existing data
            json.dump({"dim": dim,
                       "loss": loss,
                       "auc": auc,
                       "train_time": toc-tic,
                       "tain_epochs": ep_count.count,
                       "args": {"lr": lr,
                                "num_epochs": num_epochs,
                                "patience": patience,
                                "dist": dist}
                       }, f)
        if save_coords:
            return str(coords_file)
        else:
            return coords


if __name__ == '__main__':
    ScriptParser(train).run()
