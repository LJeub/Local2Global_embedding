import torch
import torch.utils.data
import torch.utils
from hyperopt import fmin, hp, tpe, Trials, space_eval, rand
from hyperopt.pyll import scope
import pandas as pd
import numpy as np
from math import log, log2, ceil
from copy import deepcopy
from itertools import count, chain, product, groupby
from tqdm.auto import tqdm
from collections.abc import Sequence


from local2global_embedding.utils import EarlyStopping, get_device


class Logistic(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        return self.softmax(self.linear(x))


def _mlp_hidden_layer(in_dim, out_dim, batch_norm=True, dropout=0, relu_last=False):
    lin = torch.nn.Linear(in_dim, out_dim, bias=True)
    nl = torch.nn.ReLU()

    if batch_norm:
        bn = torch.nn.BatchNorm1d(out_dim)
        if relu_last:
            layer_list = (lin, bn, nl)
        else:
            layer_list = (lin, nl, bn)
    else:
        layer_list = (lin, nl)

    if dropout > 0:
        return *layer_list, torch.nn.Dropout(dropout)
    else:
        return layer_list


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, batch_norm=False, dropout=0, relu_last=False):
        super().__init__()
        self.network = torch.nn.Sequential(*_mlp_hidden_layer(input_dim, hidden_dim, batch_norm, dropout, relu_last),
                                           *chain.from_iterable(_mlp_hidden_layer(hidden_dim, hidden_dim, batch_norm,
                                                                                  dropout, relu_last)
                                                                for _ in range(n_layers-2)),
                                           torch.nn.Linear(hidden_dim, output_dim, bias=True),
                                           torch.nn.LogSoftmax(dim=-1))
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.reset_parameters()

    def forward(self, x):
        return self.network(x)

    def reset_parameters(self):
        for layer in self.network:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

    def __repr__(self):
        return f"{self.__class__.__name__}({self.hidden_dim}, {self.n_layers})"


class SNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim, bias=False), torch.nn.SELU(),
                                           *chain.from_iterable((torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
                                                                 torch.nn.SELU()) for _ in range(n_layers - 2)),
                                           torch.nn.Linear(hidden_dim, output_dim, bias=False))
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.reset_parameters()

    def forward(self, x):
        return self.network(x)

    def reset_parameters(self):
        for layer in self.network:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')


def random_split(y, num_train_per_class=20, num_val=500):
    split = {}
    num_classes = int(y.max().item()) + 1
    train_mask = torch.zeros(y.size(), dtype=torch.bool)
    val_mask = torch.zeros(y.size(), dtype=torch.bool)
    test_mask = torch.zeros(y.size(), dtype=torch.bool)
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))]
        idx = idx[:num_train_per_class]
        train_mask[idx] = True
    split['train'] = train_mask.nonzero().view(-1)
    remaining = (~train_mask).nonzero().view(-1)
    remaining = remaining[remaining >= 0]  # only consider labelled data
    remaining = remaining[torch.randperm(remaining.size(0))]

    split['val'] = remaining[:num_val]
    split['test'] = remaining[num_val:]
    return split


class ClassificationProblem:
    def __init__(self, y, x=None, split=None):
        self.y = y
        self.x = x
        self.num_labels = int(y.max()) + 1
        if split is None:
            self.resplit()
        else:
            self.split = split
        self._val_data = None
        self._test_data = None
        self._train_data = None

    def resplit(self, num_train_per_class=20, num_val=500):
        self.split = random_split(self.y, num_train_per_class=num_train_per_class, num_val=num_val)

    @property
    def split(self):
        return {'train': self.train_index, 'val': self.val_index, 'test': self.test_index}

    @property
    def num_features(self):
        if self.x is None:
            return None
        else:
            return self.x.shape[1]

    @split.setter
    def split(self, split):
        self.train_index = split['train']
        self.val_index = split['val']
        self.test_index = split['test']

    def training_data(self, include_unlabeled=False):
        if self.x is None:
            raise RuntimeError('Need to set embedding first')
        if include_unlabeled:
            y = torch.tensor(self.y)
            y[self.val_index] = -1
            y[self.test_index] = -1
            if isinstance(self.x, np.memmap):
                return MMapData(self.x, y)
            else:
                x = torch.as_tensor(self.x)
                return torch.utils.data.TensorDataset(x, y)
        else:
            if self._train_data is None:
                self._train_data = torch.utils.data.TensorDataset(torch.as_tensor(self.x[self.train_index, :]),
                                                  torch.as_tensor(self.y[self.train_index]))
            return self._train_data

    def validation_data(self):
        if self._val_data is None:
            self._val_data = torch.utils.data.TensorDataset(torch.as_tensor(self.x[self.val_index, :]),
                                              torch.as_tensor(self.y[self.val_index]))
        return self._val_data

    def test_data(self):
        if self._test_data is None:
            self._test_data = torch.utils.data.TensorDataset(torch.as_tensor(self.x[self.test_index, :]),
                                              torch.as_tensor(self.y[self.test_index]))
        return self._test_data

    def labeled_data(self):
        return torch.utils.data.TensorDataset(torch.as_tensor(self.x[self.y >= 0, :]),
                                              torch.as_tensor(self.y[self.y >= 0]))

    def all_data(self):
        if isinstance(self.x, np.memmap):
            return MMapData(self.x, self.y)
        else:
            return torch.utils.data.TensorDataset(torch.as_tensor(self.x), torch.as_tensor(self.y))


class MMapData(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return torch.as_tensor(self.x[item]), torch.as_tensor(self.y[item])


class logger:
    def __init__(self, data, model):
        self.loss = []
        self.val_loss = []
        self.test_loss = []
        self.data = data
        self.model = model
    def __call__(self, l):
        self.loss.append(l)
        self.val_loss.append(accuracy(self.data, self.model, mode='val'))
        self.test_loss.append(accuracy(self.data, self.model, mode='test'))


class VATloss(torch.nn.Module):
    def __init__(self, epsilon, xi=1e-6, it=1):
        super().__init__()
        self.epsilon = epsilon
        self.xi = xi
        self.it = it
        self.divergence = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, model: torch.nn.Module, x, p=None):
        if p is None:
            with torch.no_grad():
                p = model(x)
        with torch.no_grad():
            r = torch.randn_like(x)
            r /= torch.norm(r, p=2, dim=-1, keepdim=True)
            p = p.detach()
        for _ in range(self.it):
            with torch.no_grad():
                r = (self.xi * r).clone().detach()
            r.requires_grad = True
            model.zero_grad()
            d = self.divergence(model(x + r), p)
            d.backward()
            with torch.no_grad():
                r.grad += 1e-16
                r = (r.grad / torch.norm(r.grad, p=2, dim=-1, keepdim=True))
        with torch.no_grad():
            r = (self.epsilon * r).detach()
        model.zero_grad()
        div = self.divergence(model(x + r), p)
        # div = torch.sum(p * self.logsoftmax(model(x + r_adv)))
        return div


class EntMin(torch.nn.Module):
    def forward(self, logits):
        return torch.mean(torch.distributions.Categorical(logits=logits).entropy(), dim=0)


class BatchedData(torch.utils.data.Dataset):
    def __init__(self, data: torch.utils.data.TensorDataset, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __getitem__(self, item):
        index = item*self.batch_size
        return self.data[index:index+self.batch_size]

    def __len__(self):
        return len(range(0, len(self.data), self.batch_size))



def train(data: ClassificationProblem, model: torch.nn.Module, num_epochs, batch_size, lr=0.01, batch_logger=lambda loss: None,
          epoch_logger=lambda epoch: None, device=None, epsilon=1, alpha=0, beta=0, weight_decay=1e-2, decay_lr=False, xi=1e-6,
          vat_it=1,
          teacher_alpha=0, beta_1=0.9, beta_2=0.999, adam_epsilon=1e-8, patience=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if alpha > 0 or beta > 0:
        train_data = data.training_data(include_unlabeled=True)
    else:
        train_data = data.training_data(include_unlabeled=False)
    model = model.to(device)
    if teacher_alpha:
        teacher = deepcopy(model)
        for param in teacher.parameters():
            param.detach_()

        it_count = count()

        def update_teacher():
            it = next(it_count)
            alpha = min(1 - 1 / (it + 1), teacher_alpha)
            for teacher_param, model_param in zip(teacher.parameters(), model.parameters()):
                teacher_param.data.mul_(alpha).add_(model_param, alpha=1 - alpha)
    else:
        teacher = None

        def update_teacher():
            pass

    # optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta_1, beta_2),
                                 eps=adam_epsilon)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    data_loader = torch.utils.data.DataLoader(BatchedData(train_data, batch_size=batch_size), batch_size=1,
                                              shuffle=True, pin_memory=not train_data.x.is_cuda, collate_fn=lambda b: b[0])
    if decay_lr:
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(data_loader) * num_epochs)

        def step_lr():
            lr_sched.step()
    else:
        def step_lr():
            pass

    if patience is None:
        patience = float('inf')

    criterion = torch.nn.NLLLoss(reduction='mean', ignore_index=-1)
    vat_loss = VATloss(epsilon=epsilon, xi=xi, it=vat_it)
    ent_loss = EntMin()
    if alpha == 0:
        if beta == 0:
            def loss_fun(model, x, y):
                return criterion(model(x), y)
        else:
            def loss_fun(model, x, y):
                p = model(x)
                return criterion(p, y) + beta*ent_loss(p)
    else:
        if beta == 0:
            def loss_fun(model, x, y):
                p = model(x)
                return criterion(p, y) + alpha*vat_loss(model, x, p)
        else:
            def loss_fun(model, x, y):
                p = model(x)
                return criterion(p, y) + alpha*vat_loss(model, x, p) + beta*ent_loss(p)

    x_val, y_val = data.validation_data()[:]
    x_val = x_val.to(device=device, dtype=torch.float32)
    y_val = y_val.to(device=device)
    with EarlyStopping(patience) as stop:
        with tqdm(total=num_epochs, desc='training epoch') as progress:
            for e in range(num_epochs):
                model.train()
                for x, y in data_loader:
                    x = x.to(device=device, dtype=torch.float32, non_blocking=True).view(-1, x.size(-1))
                    y = y.to(device=device, non_blocking=True).view(-1)
                    optimizer.zero_grad()
                    loss = loss_fun(model, x, y)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()
                    step_lr()
                    update_teacher()
                    batch_logger(float(loss))
                epoch_logger(e)
                model.eval()
                vl = criterion(model(x_val), y_val)
                progress.update()
                # progress.write(f'validation loss: {vl}')
                if stop(vl, model):
                    print(f'early stopping at epoch {e}')
                    break
    return model


def predict(x, model: torch.nn.Module):
    eval_state = model.training
    model.eval()
    x = x.to(device=get_device(model), dtype=torch.float32)
    with torch.no_grad():
        labels = torch.argmax(model(x), dim=-1)
    model.training = eval_state
    return labels


def accuracy(data: ClassificationProblem, model: torch.nn.Module, mode='test', batch_size=None):
    if mode == 'test':
        data = data.test_data()
    elif mode == 'val':
        data = data.validation_data()
    elif mode == 'all':
        data = data.labeled_data()
    else:
        raise ValueError(f'unknown mode {mode}')
    if batch_size is None:
        batch_size = len(data)
    loader = torch.utils.data.DataLoader(data, batch_size)
    val = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=get_device(model), dtype=torch.float32)
            y = y.to(get_device(model))

            val += torch.sum(predict(x, model) == y).cpu().item()
    return val / len(data)


def validation_accuracy(data, model: torch.nn.Module, batch_size=None):
    return accuracy(data, model, mode='val', batch_size=batch_size)


class HyperTuneObjective:
    def __init__(self, data, n_tries=1, **kwargs):
        self.data = data
        self.args = kwargs
        self.min_loss = float('inf')
        self.best_parameters = deepcopy(self.args)
        self.n_tries = n_tries

    def __call__(self, args):
        cum_loss = 0
        for _ in range(self.n_tries):
            model = train(self.data, **args, **self.args)
            loss = 1 - validation_accuracy(self.data, model)
            cum_loss += loss
            if loss < self.min_loss:
                self.min_loss = loss
                self.best_parameters.update(deepcopy(args))
            model.reset_parameters()
        return cum_loss / self.n_tries


@scope.define
def mlp_model(*args, **kwargs):
    return MLP(*args, **kwargs)


@scope.define
def linear_model(in_dim, out_dim):
    return torch.nn.Linear(in_dim, out_dim)


@scope.define
def snn_model(in_dim, hidden_dim, out_dim, n_layers):
    return SNN(in_dim, hidden_dim, out_dim, n_layers)


def grid_search(data, param_grid, epochs=10, batch_size=100, param_transform=lambda args: args, **kwargs):
    objective = HyperTuneObjective(data, **kwargs)
    results = []
    total = 1
    for v in param_grid.values():
        total *= len(v)

    for params in tqdm(product(*param_grid.values()), total=total, desc='grid search'):
        args = dict(zip(param_grid.keys(), params))
        args['model'].reset_parameters()
        args['loss'] = objective(args)
        args = param_transform(args)
        results.append(args)
    return objective.best_model, objective.best_parameters, pd.DataFrame.from_records(results)


def _make_space(args):
    log_vars = {'epsilon', 'weight_decay', 'xi', 'alpha', 'beta', 'lr'}
    int_vars = {'n_layers'}
    uniform_vars = {'dropout'}
    choice_vars = {'hidden', 'batch_norm'}
    space = {}
    fixed = {}
    for key, val in args.items():
        if isinstance(val, Sequence):
            if key in log_vars:
                space[key] = hp.loguniform(key, log(min(val)), log(max(val)))
            if key in int_vars:
                space[key] = hp.uniformint(key, min(val), max(val))
            if key in uniform_vars:
                space[key] = hp.uniform(key, min(val), max(val))
            if key in choice_vars:
                space[key] = hp.choice(key, val)
        else:
            space[key] = val
    return space


def hyper_tune(data: ClassificationProblem, max_evals=100, n_tries=1, random_search=False,
               model_args=None, train_args=None):
    _model_args = {'hidden_dim': (128, 256, 512, 1024), 'n_layers': (2, 4), 'dropout': (0, 1), 'batch_norm': (False, True)}
    _train_args = {'batch_size': 100000, 'num_epochs': 1000, 'patience': 20, 'lr': (1e-4, 1e-1)}
    model_space = _make_space(_model_args)
    train_space = _make_space(_train_args)
    objective = HyperTuneObjective(data, n_tries=n_tries)
    trials = Trials()
    in_dim = data.num_features
    out_dim = data.num_labels

    search_space = {
        **train_space,
        'model': scope.mlp_model(in_dim=in_dim,
                                 out_dim=out_dim,
                                 **model_space,
                                 )
    }

    def transform(space, value):
        value = space_eval(space, {key: val[0] for key, val in value.items() if val})
        if isinstance(value, torch.nn.Module):
            value = repr(value)
        return value

    if random_search:
        args = fmin(fn=objective, space=search_space, algo=rand.suggest, max_evals=max_evals, trials=trials)
    else:
        args = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    params = {**train_space, **model_space}
    results = pd.DataFrame.from_records({key: transform(params[key], t['misc']['vals'])
                                         for key in t['misc']['vals']}
                                        for t in trials.trials)
    results['loss'] = trials.losses()
    best_args = space_eval(params, args)
    best_model_args = {key: best_args[key] for key in _model_args}
    best_train_args = {key: best_args[key] for key in _train_args}
    return best_model_args, best_train_args, results


def plot_hyper_results(results, plot_kws=None, diag_kws=None, **kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plot_kws = {'size': 3, **(plot_kws if plot_kws is not None else {})}
    diag_kws = {'multiple': 'stack', **(diag_kws if diag_kws is not None else {})}

    if 'model' in results.columns:
        def key_fun(x):
            return [int("".join(val)) if key else "".join(val) for key, val in groupby(x, key=lambda x: x.isdigit())]

        results.model = pd.Categorical(results.model)
        results.model = results.model.cat.reorder_categories(sorted(results.model.cat.categories, key=key_fun),
                                                             ordered=True)
        f = sns.PairGrid(results, y_vars='loss', hue='model', **kwargs)
    else:
        f = sns.PairGrid(results, y_vars='loss', **kwargs)

    log_axes = {'epsilon', 'lr', 'weight_decay', 'alpha', 'xi', 'adam_epsilon', 'beta'}
    label_map = {'epsilon': r'$\epsilon$', 'lr': r'$\lambda$', 'weight_decay': r'$\omega$', 'alpha': r'$\alpha$', 'xi': r'$\xi$',
                 'hidden': r'$h$', 'beta': r'$\beta$', 'loss': 'validation error'}
    log2_axes = {}
    cat_axes = {'hidden'}

    def plot_fun(*args, **kwargs):
        ax = plt.gca()
        label = ax.get_xlabel()

        if label in cat_axes:
            sns.swarmplot(*args, **kwargs)
        else:
            kwargs = kwargs.copy()
            size = kwargs.get('s', kwargs['size'])
            kwargs['s'] = size ** 2
            del kwargs['size']
            sns.scatterplot(*args, **kwargs)


        # if ax.get_xlabel() == 'hidden':
        #     ax.set_xticks(hidden_vals)
        #     ax.set_xticklabels(hidden_vals)
        #     ax.set_xlim(0.9 * hidden_vals.min(), hidden_vals.max() / 0.9)
        #     ax.minorticks_off()
    f.map_offdiag(plot_fun, **plot_kws)
    f.map_diag(sns.histplot, **diag_kws)
    for axs in f.axes:
        for ax in axs:
            xlabel = ax.get_xlabel()
            if xlabel in log_axes:
                ax.set_xscale('log')
            if xlabel in log2_axes:
                ax.set_xscale('log', base=2)
                ax.xaxis.set_major_formatter('{x:.0f}')
            if xlabel in label_map:
                ax.set_xlabel(label_map[xlabel])

            ylabel = ax.get_ylabel()
            if ylabel in label_map:
                ax.set_ylabel(label_map[ylabel])
    return f
