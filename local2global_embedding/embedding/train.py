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
import torch


def lr_grid_search(data, model, loss_fun, validation_loss_fun, lr_grid=(0.1, 0.01, 0.005, 0.001),
                   num_epochs=10, runs=1, verbose=True):
    """
    grid search over learning rate values

    Args:
        data: input data
        model: model to train
        loss_fun: training loss takes model and data as input
        validation_loss_fun: function to compute validation loss input: (model, data)
        lr_grid: learning rate values to try
        num_epochs: number of epochs for training
        runs: number of training runs to average over for selecting best learning rate
        verbose: if ``True``, output training progress

    Returns:
        best learning rate, validation loss for all runs
    """
    val_loss = torch.zeros((len(lr_grid), runs))
    val_start = torch.zeros((len(lr_grid), runs))
    for i, lr in enumerate(lr_grid):
        for r in range(runs):
            model.reset_parameters()
            model = train(data, model, loss_fun, num_epochs=num_epochs, lr=lr, verbose=verbose)
            val_loss[i, r] = validation_loss_fun(model, data)
    model.reset_parameters()
    return lr_grid[torch.argmax(torch.mean(val_loss, 1))], val_loss


def train(data, model, loss_fun, num_epochs=100, verbose=True, lr=0.01, logger=lambda loss: None):
    """
    train an embedding model

    Args:
        data: network data
        model: embedding auto-encoder model
        loss_fun: loss function to use with model (takes arguments ``model``, ``data``)
        num_epochs: number of training epochs
        verbose: if ``True``, display training progress (default: ``True``)
        lr: learining rate (default: 0.01)
        logger: function that receives the training loss as input and is called after each epoch (does nothing by default)

    Returns:
        trained model

    This function uses the Adam optimizer for training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_fun(model, data)
        loss.backward()
        optimizer.step()
        logger(float(loss))
        if verbose:
            print(f'epoch {e}: loss={loss.item()}')
        # schedule.step()
    return model
