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

import umap
import umap.plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import torch

from local2global_embedding.run.utils import ScriptParser, load_classification_problem


rng = np.random.default_rng()


def plot_embedding(filename, name, mmap_mode=None, max_points=50000, restrict_lcc=False, size=2.0, dpi=300,
                   shader_dpi=75):
    filename = Path(filename)
    cl = load_classification_problem(name, restrict_lcc=restrict_lcc)
    fig = plt.figure(figsize=(size, size), dpi=shader_dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    y = np.asanyarray(cl.y)
    nodes = np.flatnonzero(y >= 0)
    if len(nodes) > max_points:
        nodes = rng.choice(nodes, size=(max_points,), replace=False)
    if filename.suffix == '.pt':
        coords = np.asanyarray(torch.load(filename, map_location='cpu'))
    else:
        coords = np.load(filename, mmap_mode=mmap_mode)[nodes]
    mapper = umap.UMAP().fit(coords)
    num_labels = y.max() + 1
    if num_labels <= 8:
        colors = plt.get_cmap('Set2')
        color_key = {i: matplotlib.colors.to_hex(colors.colors[i]) for i in range(num_labels)}
        color_key_cmap = None
    else:
        color_key_cmap = 'plasma'
        color_key = None

    umap.plot.points(mapper, ax=ax, labels=y[nodes], color_key_cmap=color_key_cmap, color_key=color_key,
                     show_legend=False, width=int(size*shader_dpi),
                     height=int(size*shader_dpi))
    ax.set_axis_off()
    plt.margins(0.01, 0.01)
    for c in ax.get_children():
        if type(c) is plt.Text:
            try:
                c.remove()
            except NotImplementedError:
                pass
    plt.savefig(filename.with_suffix('.png'), dpi=dpi)


if __name__ == '__main__':
    ScriptParser(plot_embedding).run()
