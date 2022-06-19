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

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from local2global_embedding.run.utils import ScriptParser, ResultsDict


def _extract_error(data, key):
    err = None
    if key == 'acc_mean':
        err = data.get('acc_std')
    if err is not None:
        err = np.asarray(err).flatten()
    return err


def plot(data, key, baseline_data=None, nt_data=None):
    fig = plt.figure()
    bar_opts = dict(elinewidth=0.5, capthick=0.5, capsize=3)
    if baseline_data is not None and key in baseline_data:
        _, cap, ebar = plt.errorbar(baseline_data['dims'], baseline_data[key],
                     yerr=_extract_error(baseline_data, key),
                     label='full', marker='o', color='tab:blue', **bar_opts)
        if ebar:
            cap[0].set_alpha(0.5)
            cap[1].set_alpha(0.5)
            ebar[0].set_alpha(0.5)


    _, cap, ebar = plt.errorbar(data['dims'], data[key], fmt='--', yerr=_extract_error(data, key),
                 label='l2g', marker='>', color='tab:blue', **bar_opts)

    if ebar:
        cap[0].set_alpha(0.5)
        cap[1].set_alpha(0.5)
        ebar[0].set_alpha(0.5)
        ebar[0].set_linestyle('--')

    if nt_data is not None and key in nt_data:
        _, cap, ebar = plt.errorbar(nt_data['dims'], nt_data[key], fmt=':', yerr=_extract_error(nt_data, key),
                     label='no-trans', color='tab:blue', linewidth=1, **bar_opts)
        if ebar:
            cap[0].set_alpha(0.5)
            cap[1].set_alpha(0.5)
            ebar[0].set_alpha(0.5)
            ebar[0].set_linestyle(':')

    plt.xscale('log')
    plt.xticks(data['dims'], data['dims'])
    plt.minorticks_off()
    if key == 'auc':
        plt.ylim(0.48, 1.02)
    plt.xlabel('embedding dimension')
    if key == 'auc':
        plt.ylabel('AUC')
    elif key == 'acc_mean':
        plt.ylabel('classification accuracy')
    plt.legend()
    return fig


def plot_all(folder=None):
    if folder is None:
        folder = Path.cwd()
    else:
        folder = Path(folder)

    for file in folder.glob('**/**/*_l2g_*_eval.json'):
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

        nt = file.with_name(base_name.replace('_l2g_', '_nt_').replace('_scale', ''))
        if nt.is_file():
            with open(nt) as f:
                nt_data = json.load(f)
        else:
            nt_data = None
        name = file.stem.split('_', 1)[0]
        network_data = torch.load(folder / f'{name}_data.pt')
        all_edges = network_data.num_edges
        patch_files = list(file.parents[1].glob('patch*_data.pt'))
        patch_edges = sum(torch.load(patch_file, map_location='cpu').num_edges
                          for patch_file in patch_files)
        oversampling_ratio = patch_edges / all_edges
        num_labels = network_data.y.max().item() + 1
        title = f"oversampling ratio: {oversampling_ratio:.2}, #patches: {len(patch_files)}"
        if 'auc' in data:
            fig = plot(data, 'auc', baseline_data, nt_data)
            ax = fig.gca()
            ax.set_title(title)
            fig.savefig(file.with_name(file.name.replace('.json', '_auc.pdf')))

        if 'acc_mean' in data:
            fig = plot(data, 'acc_mean', baseline_data, nt_data)
            ax = fig.gca()
            ax.set_title(title)
            ax.set_ylim(0.98/num_labels, 1.02)
            fig.savefig(file.with_name(file.name.replace('.json', '_cl.pdf')))


if __name__ == '__main__':
    ScriptParser(plot_all).run()
