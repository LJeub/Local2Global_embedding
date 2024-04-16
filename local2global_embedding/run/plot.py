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
from statistics import mean, stdev

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from local2global_embedding.run.utils import ScriptParser, ResultsDict, load_data
from local2global_embedding.utils import flatten


def _extract_error(data, key):
    err = None
    if key == 'acc_mean':
        err = data.get('acc_std')
    if err is not None:
        err = np.asarray(err).flatten()
    return err


def _normalise_data(data):
    data = np.asarray(data)
    data = data.flatten()
    return data


def mean_and_deviation(data):
    data = [flatten(v) for v in data]
    data_mean = [mean(v) for v in data]
    data_std = [stdev(v) for v in data]
    return data_mean, data_std


def plot_with_errorbars(x, y_mean, y_err, fmt='-', **kwargs):
    opts = dict(elinewidth=0.5, capthick=0.5, capsize=3)
    opts["fmt"] = fmt
    opts.update(kwargs)
    _, cap, ebar = plt.errorbar(_normalise_data(x), _normalise_data(y_mean),
                                yerr=_normalise_data(y_err), **opts)
    if ebar:
        cap[0].set_alpha(0.5)
        cap[1].set_alpha(0.5)
        ebar[0].set_alpha(0.5)
        ebar[0].set_linestyle(fmt)


def plot(data, key, baseline_data=None, nt_data=None, rotate_data=None, translate_data=None):
    fig = plt.figure()
    if baseline_data is not None and key in baseline_data:
        d_mean, d_err = mean_and_deviation(baseline_data[key])
        plot_with_errorbars(baseline_data['dims'], d_mean, d_err, label='full', marker='o', color='tab:blue', zorder=4)

    d_mean, d_err = mean_and_deviation(data[key])
    plot_with_errorbars(data['dims'], d_mean, d_err, fmt='-',
                 label='l2g', marker='>', color='tab:red', zorder=5)

    if rotate_data is not None and key in rotate_data:
        d_mean, d_err = mean_and_deviation(rotate_data[key])
        plot_with_errorbars(rotate_data['dims'], d_mean, d_err, fmt='--', marker='s', markersize=3,
                     label='rotate-only', color='tab:orange', linewidth=0.5, zorder=3)

    if translate_data is not None and key in translate_data:
        d_mean, d_err = mean_and_deviation(translate_data[key])
        plot_with_errorbars(translate_data['dims'], d_mean, d_err, fmt='-.', marker='d', markersize=3,
                     label='translate-only', color='tab:purple', linewidth=0.5, zorder=2)

    if nt_data is not None and key in nt_data:
        d_mean, d_err = mean_and_deviation(nt_data[key])
        plot_with_errorbars(nt_data['dims'], d_mean, d_err, fmt=':',
                            label='no-l2g', color='tab:pink', linewidth=0.5, zorder=1)

    plt.xscale('log')
    plt.xticks(data['dims'], data['dims'])
    plt.minorticks_off()
    if key == 'auc':
        plt.ylim(0.48, 1.02)
    plt.xlabel('embedding dimension')
    if key == 'auc':
        plt.ylabel('AUC')
    elif key == 'acc':
        plt.ylabel('classification accuracy')
    plt.legend(ncol=3, frameon=False)
    return fig


def plot_all(folder=None):
    """
    Plot results

    Args:
        folder: results folder (default: CWD)
    """
    if folder is None:
        folder = Path.cwd()
    else:
        folder = Path(folder)

    for file in folder.glob('**/**/*_l2g_scale_eval.json'):
        print(file)
        with open(file) as f:
            data = json.load(f)
        base_name_parts = file.name.split('_hc', 1)
        if len(base_name_parts) > 1:
            base_name = base_name_parts[0] + '_' + base_name_parts[1].split('_', 1)[1]
        else:
            base_name = base_name_parts[0]

        baseline = folder / file.parent.name / base_name.replace('_l2g_', '_full_').replace('_scale', '')
        if baseline.is_file():
            baseline_data = ResultsDict(baseline)
            baseline_data.reduce_to_dims(data['dims'])
        else:
            baseline_data = None

        nt = file.with_name(base_name.replace('_scale_', '_norotate_notranslate_'))
        if nt.is_file():
            with open(nt) as f:
                nt_data = json.load(f)
        else:
            nt_data = None

        rotate = file.with_name(base_name.replace('_scale_', '_notranslate_'))
        if rotate.is_file():
            with open(rotate) as f:
                rotate_data = json.load(f)
        else:
            rotate_data = None

        translate = file.with_name(base_name.replace('_scale_', "_norotate_"))
        if translate.is_file():
            with open(translate) as f:
                translate_data = json.load(f)
        else:
            translate_data = None

        name = file.stem.split('_', 1)[0]
        network_data = load_data(name)
        all_edges = network_data.num_edges
        patch_files = list(file.parents[1].glob('patch*_index.npy'))
        patch_edges = sum(network_data.subgraph(np.load(patch_file)).num_edges
                          for patch_file in patch_files)
        oversampling_ratio = patch_edges / all_edges
        num_labels = network_data.y.max().item() + 1
        title = f"oversampling ratio: {oversampling_ratio:.2}, #patches: {len(patch_files)}"
        if 'auc' in data:
            fig = plot(data, 'auc', baseline_data, nt_data, rotate_data, translate_data)
            ax = fig.gca()
            ax.set_title(title)
            ax.set_ylim(0.48, 1.02)
            fig.savefig(file.with_name(file.name.replace('.json', '_auc.pdf')))

        if 'acc' in data:
            fig = plot(data, 'acc', baseline_data, nt_data, rotate_data, translate_data)
            ax = fig.gca()
            ax.set_title(title)
            ax.set_ylim(0.98/num_labels, 1.02)
            fig.savefig(file.with_name(file.name.replace('.json', '_cl.pdf')))


if __name__ == '__main__':
    ScriptParser(plot_all).run()
