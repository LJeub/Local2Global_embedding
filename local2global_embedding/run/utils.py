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
from collections import UserDict
import argparse
from inspect import signature
import typing
import time
from ast import literal_eval
from traceback import print_exception
from runpy import run_path
import operator

import torch
from docstring_parser import parse as parse_doc
from filelock import SoftFileLock
from atomicwrites import atomic_write
from dask.distributed import Client, get_client

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


def load_data(name, root='/tmp', restrict_lcc=False, **kwargs):
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

    return data


def load_classification_problem(name, graph, root='/tmp', **class_args):
    root = Path(root).expanduser()
    y, split = _classification_loader[name](root=root, **class_args)
    if graph.has_node_labels():
        index = graph.nodes
        index_map = torch.full(y.shape, -1, dtype=torch.long)
        index_map[index] = torch.arange(len(index), dtype=torch.long)
        y = y[index]
        for key, value in split.items():
            mapped_index = index_map[value]
            split[key] = mapped_index[mapped_index >= 0]
    return ClassificationProblem(y, split=split)


load_data.__doc__ = load_data.__doc__.format(names=list(_dataloaders.keys()))


def cluster_string(cluster='metis', num_clusters=10, num_iters: int=None, beta=0.1, levels=1):
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
    if levels > 1:
        cluster_string += f'_hc{levels}'
    return cluster_string


def patch_folder_name(name: str, min_overlap: int, target_overlap: int, cluster='metis',
                      num_clusters=10, num_iters: int=None, beta=0.1, levels=1,
                      sparsify='resistance', target_patch_degree=4.0, gamma=0.0):
    if sparsify == 'resistance':
        sp_string = f"resistance_deg{target_patch_degree}"
    elif sparsify == 'rmst':
        sp_string = f"rmst_gamma{gamma}"
    elif sparsify == 'none':
        sp_string = "no_sparsify"
    elif sparsify == 'sample':
        sp_string = f'sample_deg{target_patch_degree}'
    elif sparsify == 'neighbors':
        sp_string = f'neighbors_deg{target_patch_degree}'
    else:
        raise RuntimeError(f"Unknown sparsification method '{sparsify}'.")
    cl_string = cluster_string(cluster, num_clusters, num_iters, beta, levels)

    return f'{name}_{cl_string}_{sp_string}_mo{min_overlap}_to{target_overlap}_patches'


def cluster_file_name(name, cluster='metis', num_clusters=10, num_iters: int=None, beta=0.1, levels=1):
    cl_string = cluster_string(cluster, num_clusters, num_iters, beta, levels)
    return f'{name}_{cl_string}_clusters.pt'


class NoLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def acquire(self):
        pass

    def release(self):
        pass


class SyncDict(UserDict):
    """
    Class for keeping json-backed dict with locking for sync
    """
    def load(self):
        """
        restore results from file

        Returns:
            populated ResultsDict

        """
        with self._lock:
            with open(self.filename) as f:
                self.data = json.load(f)

    def save(self):
        """
        dump contents to json file

        Args:
            filename: output file path

        """
        with self._lock:
            with atomic_write(self.filename, overwrite=True) as f:  # this should avoid any chance of loosing existing data
                json.dump(self.data, f)

    def __init__(self, filename, lock=True):
        """
        initialise empty ResultsDict
        Args:
            filename: file to use for storage
            lock: set lock=False to avoid locking (use wisely)
        """
        super().__init__()
        self.filename = Path(filename)
        if lock:
            self._lock = SoftFileLock(self.filename.with_suffix('.lock'), timeout=10)
        else:
            self._lock = NoLock()  # implements lock interface without doing anything
        with self._lock:
            if not self.filename.is_file():
                self.save()
            else:
                self.load()

    def __enter__(self):
        self._lock.acquire()
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
        self._lock.release()


class ResultsDict(UserDict):
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
                self.data = json.load(f)

    def save(self):
        """
        dump contents to json file

        Args:
            filename: output file path

        """
        with self._lock:
            with atomic_write(self.filename, overwrite=True) as f:  # this should avoid any chance of loosing existing data
                json.dump(self.data, f)

    def __init__(self, filename, replace=False, lock=True):
        """
        initialise empty ResultsDict
        Args:
            replace: set the replace attribute (default: ``False``)
        """
        super().__init__()
        self.filename = Path(filename)
        if lock:
            self._lock = SoftFileLock(self.filename.with_suffix('.lock'), timeout=10)
        else:
            self._lock = NoLock()  # implements lock interface without doing anything
        with self._lock:
            if not self.filename.is_file():
                self.data = {'dims': [], 'runs': []}
                self.save()
            else:
                self.load()
        self.replace = replace  #: if ``True``, updates replace existing data, if ``False``, updates append data

    def __enter__(self):
        self._lock.acquire()
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
        self._lock.release()

    def _update_index(self, index, replace=False, **kwargs):
        """
        update data for a given index

        Args:
            index: integer index into data lists
            aucs: new auc value
            args: new args data (optional)

        """
        for key, val in kwargs.items():
            if key not in self:
                self.data[key] = [[] for _ in self['dims']]
            if replace or self.replace:
                self[key][index] = [val]
            else:
                self[key][index].append(val)
        if not (replace or self.replace):
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
        for key in self:
            if key not in {'dims', 'runs'}:  # these only store single values per dimension
                self[key].insert(index, [])
        for key, val in kwargs.items():
            if key in self:
                self[key][index].append(val)
            else:
                self.data[key] = [[] for _ in self['dims']]
                self[key][index].append(val)
        self['runs'].insert(index, 1)

    def _index(self, dim):
        return bisect_left(self['dims'], dim)

    def update_dim(self, dim, replace=False, **kwargs):
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
        index = self._index(dim)
        if index < len(self['dims']) and self['dims'][index] == dim:
            self._update_index(index, replace=replace, **kwargs)
        else:
            self._insert_index(index, dim, **kwargs)

    def delete_dim(self, dim):
        index = self._index(dim)
        if index < len(self['dims']) and self['dims'][index] == dim:
            for v in self.values():
                del v[index]

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

    def get(self, item, dim=None, default=None):
        if item in self:
            if dim is None:
                return self[item]
            elif self.contains_dim(dim):
                index = self._index(dim)
                return self[item][index]
        return default

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
        for key1 in self.data:
            if isinstance(self.data[key1], list):
                self.data[key1] = [self[key1][i] for i in index]
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
        await _task_queue.put(args)  # limit number of simultaneous tasks
    if _throttler is not None:
        await _throttler.submit_ok()  # limit task creation frequency
    if _stderr:
        stdout = sys.stderr  # redirect all output to stderr
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

    def __call__(self, value: str):
        if value == 'None' and type(None) in self.types:
            return None

        for t in self.types:
            try:
                return t(value)
            except Exception:
                pass
        raise RuntimeError(f'Cannot parse argument {value}')


class Argument:
    """
    Argument wrapper for ScriptParser
    """
    def __init__(self, name='', parameter=None, allow_reset=False):
        """
        Initialize Argument

        Args:
            name: argument name
            parameter: signature parameter (optional, used to get default value if specified)
        """
        self.name = name
        self.allow_reset=allow_reset
        self.required = parameter is None or parameter.default is parameter.empty
        self.is_set = False
        if not self.required:
            self._value = parameter.default
        else:
            self._value = None

    def __call__(self, input_str):
        """
        parse argument string

        Args:
            input_str: string to evaluate

        Returns: self

        Tries to parse `input_str` as python code using `ast.literal_eval`. If this fails, sets value to `input_str`.

        """
        if not self.is_set or self.allow_reset:
            self.is_set = True
            try:
                val = literal_eval(input_str)
            except Exception:  # could not interpret as python literal, assume it is a bare string argument
                val = input_str
            self._value = val
            return self
        else:
            raise RuntimeError(f"Tried to set value for argument {self.name!r} multiple times, old value: {self._value}, new value: {input_str}")

    @property
    def value(self):
        if not self.required or self.is_set:
            return self._value
        else:
            raise RuntimeError(f"Missing value for required argument {self.name!r}")

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(name={self.name!r})'
        if self.required:
            repr_str += ', required'

        if self.is_set:
            repr_str += f', value={self.value!r}'
        elif not self.required:
            repr_str += f', default={self.value!r}'

        return repr_str


class ScriptParser:
    """
    Build a command-line interface to a python function

    Inspects the function signature to create command-line arguments. It converts
    argument `arg` to  long option `--arg`. Parsing is similar to python, supporting
    mix of positional and named arguments (note its possible to specify named arguments before positional arguments).
    Also supports use of `*args` and `**kwargs`.

    Help messages are constructed by parsing the doc-string of the wrapped function.

    Can be used as a decorator if function should only be used as a script

    """
    def __init__(self, func, allow_reset=False):
        """
        Wrap `func` as a command-line interface

        Args:
            func: Callable
        """
        self.func = func
        self.parser = argparse.ArgumentParser(prog=func.__name__)
        self.allow_reset = allow_reset
        self.var_pos = False
        self.var_keyword = False
        self.arguments = []
        sig = signature(func).parameters
        docstring = parse_doc(func.__doc__)
        help = {p.arg_name: p.description for p in docstring.params}
        self.parser.description = docstring.short_description
        self.parser.add_argument('_pos', nargs='*')
        for name, parameter in sig.items():
            if parameter.kind == parameter.VAR_POSITIONAL:
                if not self.var_pos:
                    self.var_pos = True
                else:
                    raise RuntimeError('Only expected a single *args')
            elif parameter.kind == parameter.VAR_KEYWORD:
                if not self.var_keyword:
                    self.var_keyword = True
                else:
                    raise RuntimeError('Only expected a single **kwargs')
            else:
                arg = Argument(name, parameter, allow_reset)
                self.arguments.append(arg)
                if arg.required:
                    self.parser.add_argument(f'--{name}', type=arg, default=arg, help=help.get(name, name))
                else:
                    help_str = f'{help.get(name,  name)} (default: {arg.value!r})'
                    self.parser.add_argument("--{}".format(name), type=arg, default=arg, help=help_str)

    def parse(self, args=None):
        if args is None:
            args = sys.argv[1:]

        if self.var_keyword:
            arg_res, unknown = self.parser.parse_known_args(args)
            kwargs = vars(arg_res)
            unknown_parser = argparse.ArgumentParser()
            for arg in unknown:
                if arg.startswith("--"):
                    name = arg[2:].split('=', 1)[0]
                    new_arg = Argument(name, allow_reset=self.allow_reset)
                    unknown_parser.add_argument(f'--{name}', type=new_arg, default=new_arg)
                    self.arguments.append(new_arg)
            kwargs.update(vars(unknown_parser.parse_args(unknown)))
        else:
            kwargs = vars(self.parser.parse_args(args))

        args = []
        pos_args = kwargs.pop('_pos')
        for arg, val in zip(self.arguments, pos_args):
            if val.startswith('--'):
                raise RuntimeError(f'Unknown keyword argument {val}')
            if not arg.is_set:
                arg = arg(val)
                args.append(arg.value)
                kwargs.pop(arg.name)
            else:
                break

        if self.var_pos:
            args.extend(Argument()(val).value for val in pos_args[len(args):])
        else:
            if len(args) != len(pos_args):
                raise RuntimeError(f'Too many positional arguments specified. {pos_args=}')

        for name, value in kwargs.items():
            kwargs[name] = value.value

        return args, kwargs

    def run(self, args=None):
        """
        run the wrapped function with arguments passed on sys.argv or as list of string arguments
        """
        args, kwargs = self.parse(args)
        self.func(*args, **kwargs)

    def __call__(self, args=None):
        self.run(args)


def watch_progress(tasks):
    for c in tasks:
        if c.status == 'error':
            print(f'{c} errored')
            e = c.exception()
            print_exception(type(e), e, c.traceback())
        else:
            print(f'{c} complete')
        del c


def get_or_init_client(init_cluster=True):
    try:
        client = get_client()
    except ValueError:
        if init_cluster:
            print('setting up cluster')
            cluster_init_path = Path().home() / '.config' / 'dask' / 'cluster_init.py'
            kwargs = run_path(cluster_init_path)
            client = Client(kwargs['cluster'])
        else:
            print('launching default client')
            client = Client()
    return client


def dask_unpack(client, result_future, n):
    """
    Unpack dask future of iterable

    Args:
        client: dask client
        result_future: future to unpack
        n: number of items to unpack

    Returns:
        list of unpacked futures

    """
    return [client.submit(operator.getitem, result_future, i) for i in range(n)]


