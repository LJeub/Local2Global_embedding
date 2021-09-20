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
from bisect import bisect_left
from pathlib import Path
import asyncio
import sys
from collections.abc import Iterable
import argparse
from inspect import signature
import typing
import time

import torch
from docstring_parser import parse as parse_doc
from filelock import SoftFileLock

from local2global_embedding.classfication import ClassificationProblem

_dataloaders = {}  #: dataloaders
_classification_loader = {}


def dataloader(name):
    """
    decorator for registering dataloader functions

    Args:
        name: data set name

    """
    def loader(func):
        _dataloaders[name] = func
        return func
    return loader


def classificationloader(name):
    """
    decorator for registering classification data loaders
    Args:
        name: data set name

    Returns:

    """
    def loader(func):
        _classification_loader[name] = func
        return func
    return loader


def load_data(name, root='/tmp', normalise=False, restrict_lcc=False, **kwargs):
    """
    load data set

    Args:
        name: name of data set (one of {names})
        root: root dir to store downloaded data (default '/tmp')

    Returns:
        graph data

    """
    root = Path(root).expanduser()
    data = _dataloaders[name](root, **kwargs)

    if restrict_lcc:
        data = data.lcc(relabel=True)

    if normalise:
        r_sum = data.x.sum(dim=1)
        r_sum[r_sum == 0] = 1.0  # avoid division by zero
        data.x /= r_sum[:, None]

    return data


def load_classification_problem(name, root='/tmp', restrict_lcc=False, graph_args={}, class_args={}):
    root = Path(root).expanduser()
    y, split = _classification_loader[name](root=root, **class_args)
    if restrict_lcc:
        graph = load_data(name, root, restrict_lcc=False, **graph_args)
        index = graph.nodes_in_lcc()
        index_map = torch.full(y.shape, -1, dtype=torch.long)
        index_map[index] = torch.arange(len(index), dtype=torch.long)
        y = y[index]
        for key, value in split.items():
            mapped_index = index_map[value]
            split[key] = mapped_index[mapped_index >= 0]
    return ClassificationProblem(y, split=split)


load_data.__doc__ = load_data.__doc__.format(names=list(_dataloaders.keys()))


def patch_folder_name(name: str, min_overlap: int, target_overlap: int, cluster='metis',
                      num_clusters=10, num_iters: int=None, beta=0.1,
                      sparsify='resistance', target_patch_degree=4.0, gamma=0.0):
    if sparsify == 'resistance':
        sp_string = f"resistance_deg{target_patch_degree}"
    elif sparsify == 'rmst':
        sp_string = f"rmst_gamma{gamma}"
    elif sparsify == 'none':
        sp_string = "no_sparsify"
    elif sparsify == 'sample':
        sp_string = 'sample'
    else:
        raise RuntimeError(f"Unknown sparsification method '{sparsify}'.")

    if cluster == 'louvain':
        cluster_string = 'louvain'
    elif cluster == 'distributed':
        cluster_string = f'distributed_beta{beta}_it{num_iters}'
    elif cluster == 'fennel':
        cluster_string = f"fennel_n{num_clusters}_it{num_iters}"
    elif cluster == 'metis':
        cluster_string = f"metis_n{num_clusters}"
    else:
        raise RuntimeError(f"Unknown cluster method '{cluster}'.")
    return f'{name}_{cluster_string}_{sp_string}_mo{min_overlap}_to{target_overlap}_patches'


class ResultsDict:
    """
    Class for keeping track of results
    """
    def load(self):
        """
        restore results from file

        Args:
            filename: input json file
            replace: set the replace attribute

        Returns:
            populated ResultsDict

        """
        with self._lock:
            with open(self.filename) as f:
                self._data = json.load(f)

    def save(self):
        """
        dump contents to json file

        Args:
            filename: output file path

        """
        with self._lock:
            with open(self.filename, 'w') as f:
                json.dump(self._data, f)

    def __init__(self, filename, replace=False):
        """
        initialise empty ResultsDict
        Args:
            replace: set the replace attribute (default: ``False``)
        """
        self.filename = Path(filename)
        self._lock = SoftFileLock(self.filename.with_suffix('.lock'), timeout=10)
        with self._lock:
            if not self.filename.is_file():
                with open(self.filename, 'w') as f:
                    json.dump({'dims': [], 'runs': []}, f)
        self.replace = replace  #: if ``True``, updates replace existing data, if ``False``, updates append data
        self.load()

    def __enter__(self):
        self._lock.acquire()
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
        self._lock.release()

    def __getitem__(self, item):
        if self._data is not None:
            return self._data[item]

    def _update_index(self, index, **kwargs):
        """
        update data for a given index

        Args:
            index: integer index into data lists
            aucs: new auc value
            args: new args data (optional)

        """
        for key, val in kwargs.items():
            if key not in self:
                self._data[key] = [[] for _ in self['dims']]
            if self.replace:
                self[key][index] = [val]
            else:
                self[key][index].append(val)
        if not self.replace:
            self['runs'][index] += 1

    def _insert_index(self, index: int, dim: int, **kwargs):
        """
        insert new data at index

        Args:
            index: integer index into data lists
            dim: data dimension for index
            aucs: new auc values
            args: new args data (optional)
        """
        self['dims'].insert(index, dim)
        for key, val in kwargs.items():
            if key in self:
                self[key].insert(index, [val])
            else:
                self._data[key] = [[] for _ in self['dims']]
                self[key][index].append(val)
        self['runs'].insert(index, 1)

    def update_dim(self, dim, **kwargs):
        """
        update data for given dimension

        Args:
            dim: dimension to update
            auc: new auc value
            loss: new loss value
            args: new args data (optional)

        if ``self.contains_dim(dim) == True``, behaviour depends on the value of
        ``self.replace``

        """
        index = bisect_left(self['dims'], dim)
        if index < len(self['dims']) and self['dims'][index] == dim:
            self._update_index(index, **kwargs)
        else:
            self._insert_index(index, dim, **kwargs)

    def max(self, field, dim=None):
        """
        return maximum auc values

        Args:
            field: field to take maximum over
            dim: if ``dim=None``, return list of values for all dimension, else only return maximum value for ``dim``.

        """
        if field not in self:
            if dim is None:
                return [-float('inf') for _ in self['dims']]
            else:
                return -float('inf')
        if dim is None:
            return [max(val) for val in self[field]]
        else:
            index = bisect_left(self['dims'], dim)
            if index < len(self['dims']) and self['dims'][index] == dim:
                return max(self[field][index])
            else:
                return -float('inf')

    def min(self, field, dim=None):
        if field not in self:
            if dim is None:
                return [float('inf') for _ in self['dims']]
            else:
                return float('inf')
        if dim is None:
            return [min(val) for val in self[field]]
        else:
            index = bisect_left(self['dims'], dim)
            if index < len(self['dims']) and self['dims'][index] == dim:
                return min(self[field][index])
            else:
                return float('inf')

    def __contains__(self, item):
        return item in self._data

    def contains_dim(self, dim):
        """
        equivalent to ``dim in self['dims']``

        """
        index = bisect_left(self['dims'], dim)
        return index < len(self['dims']) and self['dims'][index] == dim

    def reduce_to_dims(self, dims):
        """
        remove all data for dimensions not in ``dims``
        Args:
            dims: list of dimensions to keep

        """
        index = [i for i, d in enumerate(dims) if self.contains_dim(d)]
        for key1 in self._data:
            if isinstance(self._data[key1], list):
                self._data[key1] = [self[key1][i] for i in index]
        return self

    def runs(self, dim=None):
        """
        return the number of runs

        Args:
            dim: if ``dim is None``, return list of number of runs for all dimension, else return number of
                 runs for dimension ``dim``.

        """
        if dim is None:
            return self['runs']
        else:
            index = bisect_left(self['dims'], dim)
            if index < len(self['dims']) and self['dims'][index] == dim:
                return self['runs'][index]
            else:
                return 0


class Throttler:
    def __init__(self, min_interval=0):
        self.min_interval=min_interval
        self.next_run = time.monotonic()

    async def submit_ok(self):
        now = time.monotonic()
        if now > self.next_run + self.min_interval:
            self.next_run = now
            return
        else:
            self.next_run += self.min_interval
            await asyncio.sleep(self.next_run-now)
            return


async def run_script(script_name, _cmd_prefix=None, _task_queue: asyncio.Queue = None, _throttler: Throttler = None,
                     _stderr=False,
                     **kwargs):
    args = []
    if _cmd_prefix is not None:
        args.extend(_cmd_prefix.split())
    args.extend(['python', '-m', f'local2global_embedding.run.scripts.{script_name}'])
    args.extend(f'--{key}={value}' for key, value in kwargs.items())
    if _task_queue is not None:
        await _task_queue.put(args)
    if _throttler is not None:
        await _throttler.submit_ok()
    if _stderr:
        stdout = sys.stderr
    else:
        stdout = None
    proc = await asyncio.create_subprocess_exec(*args, stdout=stdout)
    await proc.communicate()
    if _task_queue is not None:
        await _task_queue.get()
        _task_queue.task_done()


class CSVList:
    def __init__(self, dtype=str):
        self.dtype=dtype

    def __call__(self, input: str):
        return [self.dtype(s) for s in input.split(',')]


class BooleanString:
    def __new__(cls, s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'


class Union:
    def __init__(self, types):
        self.types = types

    def __call__(self, input: str):
        if input == 'None' and type(None) in self.types:
            return None

        for t in self.types:
            try:
                return t(input)
            except Exception:
                pass
        raise RuntimeError(f'Cannot parse argument {input}')


class ArgDefault:
    def __init__(self, value):
        self.value = value


ArgRequired = object()


class ScriptParser:
    def __init__(self, func, ignore_unknown=False, nargs=None):
        self.func = func
        self.ignore_unknown = ignore_unknown
        self.nargs = nargs
        self.parser = argparse.ArgumentParser(prog=func.__name__)
        self._var_pos_type = None
        self._var_keyword_type = None
        self._arg_names = []
        self._arg_types = []
        sig = signature(func).parameters
        docstring = parse_doc(func.__doc__)
        help = {p.arg_name: p.description for p in docstring.params}
        self.parser.description = docstring.short_description
        self.parser.add_argument('_pos', nargs='*')
        for name, parameter in sig.items():
            self._arg_names.append(name)
            p_type = self._arg_type(parameter)
            self._arg_types.append(p_type)
            if parameter.kind == parameter.VAR_POSITIONAL:
                if self._var_pos_type is None:
                    self._var_pos_type = p_type
                else:
                    raise RuntimeError('Only expected a single *args')
            elif parameter.kind == parameter.VAR_KEYWORD:
                if self._var_keyword_type is None:
                    self._var_keyword_type = p_type
                else:
                    raise RuntimeError('Only expected a single **kwargs')
            elif parameter.default is parameter.empty:
                self.parser.add_argument(f'--{name}', type=p_type, default=ArgRequired, help=help.get(name, name))
            else:
                help_str = f'{help.get(name,  name)}'
                if parameter.default is not None:
                    help_str += ' (default: %(default)s)'
                self.parser.add_argument("--{}".format(name), type=p_type, default=ArgDefault(parameter.default),
                                         help=help_str)

    def _arg_type(self, parameter):
        if parameter.annotation is parameter.empty:
            if parameter.default is parameter.empty:
                p_type = str
            else:
                p_type = type(parameter.default)
        else:
            p_type = parameter.annotation

        if p_type == bool:
            p_type = BooleanString

        if hasattr(p_type, '__origin__'):
            # typing GenericAlias
            if p_type.__origin__ is list:
                p_type = CSVList(dtype=p_type.__args__[0])
            elif p_type.__origin__ is typing.Union:
                p_type = Union(p_type.__args__)

        if p_type != str and isinstance(p_type, Iterable):
            if parameter.annotation is parameter.empty:
                l_type = type(next(iter(parameter.default)))
            else:
                l_type = type(next(iter(parameter.annotation)))
            p_type = CSVList(dtype=l_type)
        return p_type

    def parse(self, args=None):
        if args is None:
            args = sys.argv[1:]
            from_sys = True
        else:
            from_sys = False

        if self.nargs is not None:
            args = args[:self.nargs]
            if from_sys:
                sys.argv = sys.argv[self.nargs+1:]

        if self._var_keyword_type is not None:
            _, unknown = self.parser.parse_known_args(args)
            for arg in unknown:
                if arg.startswith("--"):
                    self.parser.add_argument(arg, type=self._var_keyword_type)

        if self.ignore_unknown:
            arg_res, others = self.parser.parse_known_args(args)
            if from_sys:
                sys.argv = others
        else:
            arg_res = self.parser.parse_args(args)
        kwargs = vars(arg_res)
        args = []
        pos_args = kwargs.pop('_pos')
        for name, p_type, val in zip(self._arg_names, self._arg_types, pos_args):
            arg_val = kwargs[name]
            if isinstance(arg_val, ArgDefault) or arg_val is ArgRequired:
                del kwargs[name]
                args.append(p_type(val))
            else:
                break

        if self._var_pos_type is not None:
            args.extend(self._var_pos_type(val) for val in pos_args[len(args):])
        else:
            if len(args) != len(pos_args):
                raise RuntimeError('Too many positional arguments specified.')

        for name, value in kwargs.items():
            if isinstance(value, ArgDefault):
                kwargs[name] = value.value
            if value is ArgRequired:
                raise RuntimeError(f'Missing required argument {name}')

        return args, kwargs

    def run(self):
        """
        run the wrapped function with arguments passed on sys.argv
        """
        args, kwargs = self.parse()
        self.func(*args, **kwargs)
