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
import torch
from typing import Optional

from local2global_embedding.embedding import train, VGAE, VGAE_loss, GAE, GAE_loss, DGI, DGILoss, reconstruction_auc
from local2global_embedding.utils import speye
from local2global_embedding.run.utils import ResultsDict, ScriptParser


def select_loss(model):
    if isinstance(model, VGAE):
        return VGAE_loss
    elif isinstance(model, GAE):
        return GAE_loss
    elif isinstance(model, DGI):
        return DGILoss()


def create_model(model, dim, hidden_dim, num_features, dist):
    if model == 'VGAE':
        return VGAE(dim, hidden_dim, num_features, dist)
    elif model == 'GAE':
        return GAE(dim, hidden_dim, num_features, dist)
    elif model == 'DGI':
        return DGI(num_features, dim)


def main(data, model, lr: float, num_epochs: int, patience: int, verbose: bool, results_file: str,
         dim: int, hidden_multiplier: Optional[int] = None, no_features=False, dist=False,
         device: Optional[str] = None):
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
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    data = torch.load(data).to(device)

    if no_features:
        data.x = speye(data.num_nodes).to(device)

    model = create_model(model, dim, dim * hidden_multiplier, data.num_features, dist)
    loss_fun = select_loss(model)
    model = train(data, model, loss_fun, num_epochs, patience, lr, verbose=verbose)
    coords = model.embed(data)

    auc = reconstruction_auc(coords, data, dist=dist)
    loss = float(loss_fun(model, data))
    results_file = Path(results_file)
    model_file = results_file.with_name(results_file.name.replace('_info.json', f'_d{dim}_best_model.pt'))
    coords_file = model_file.with_name(model_file.name.replace('model', 'coords'))
    with ResultsDict(results_file) as results:
        if results.min('loss', dim) > loss:
            torch.save(model.state_dict(), model_file)
            torch.save(coords, coords_file)
        results.update_dim(dim, auc=auc, loss=loss, args={'lr': lr, 'num_epochs': num_epochs,
                                                           'patience': patience, 'dist': dist})


if __name__ == '__main__':
    parser = ScriptParser(main)
    args, kwargs = parser.parse()
    main(*args, **kwargs)
