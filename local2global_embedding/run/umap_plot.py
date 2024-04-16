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
from datetime import datetime
from functools import partial

import umap
import datashader as ds
import datashader.transfer_functions as tf
from datashader.mpl_ext import dsshow, alpha_colormap

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch

from local2global_embedding.run.utils import ScriptParser, load_classification_problem


rng = np.random.default_rng()


def get_ax_size(ax):
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


def plot_embedding(filename, name, mmap_mode: Optional[str] = None, max_points=500000, restrict_lcc=False,
                   pointsize=5,
                   size=2.0, dpi=1200, data_root='/tmp', min_dist=0.0, metric='euclidean', verbose=True):
    filename = Path(filename)
    print(f'loading data started at {datetime.now()}')
    cl = load_classification_problem(name, restrict_lcc=restrict_lcc, root=data_root)
    print(f'classificaton problem loaded at {datetime.now()}')
    fig = plt.figure(figsize=(size, size), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax_size = size*dpi
    pad = 2*pointsize / ax_size
    y = np.asanyarray(cl.y)
    nodes = np.flatnonzero(y >= 0)
    if len(nodes) > max_points:
        nodes = rng.choice(nodes, size=(max_points,), replace=False)
    if filename.suffix == '.pt':
        coords = np.asanyarray(torch.load(filename, map_location='cpu'))
    else:
        coords = np.load(filename, mmap_mode=mmap_mode)[nodes]
    print(f'embedding loaded at {datetime.now()}')
    vc = umap.UMAP(min_dist=min_dist, metric=metric, verbose=verbose).fit_transform(coords)
    min_range = vc.min(axis=0)
    max_range = vc.max(axis=0)
    pad = (max_range-min_range) * pad

    x_range = (min_range[0]-pad[0], max_range[0]+pad[0])
    y_range = (min_range[1]-pad[1], max_range[1]+pad[1])

    df = pd.DataFrame(vc, columns=['x', 'y'])
    df['label'] = y[nodes]
    df['label'] = df['label'].astype('category')
    colors = sns.color_palette('husl', cl.num_labels)
    colors = {i: tuple(int(vi * 255) for vi in v) for i, v in enumerate(colors)}
    dsshow(df, ds.Point('x', 'y'), ds.count_cat('label'), ax=ax, norm='eq_hist', color_key=colors,
           shade_hook=partial(tf.dynspread, threshold=0.99, max_px=pointsize, shape='circle'), alpha_range=(55, 255),
           x_range=x_range, y_range=y_range)

    ax.set_axis_off()
    plt.margins(0.01, 0.01)
    plt.savefig(filename.with_suffix('.png'), dpi=dpi)


if __name__ == '__main__':
    ScriptParser(plot_embedding).run()
