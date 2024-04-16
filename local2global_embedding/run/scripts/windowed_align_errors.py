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

import numpy as np
from dask import delayed
from dask.distributed import secede, rejoin

from local2global.utils import SVDAlignmentProblem, local_error

from local2global_embedding.run.scripts.utils import num_nodes, num_patches


@delayed
def last_patch_error(patches, scale, use_median):
    prob = SVDAlignmentProblem(patches)
    if use_median:
        me = prob.align_patches(scale=scale).median_embedding()
    else:
        me = prob.align_patches(scale=scale).mean_embedding()
    patch = prob.patches[-1]
    return patch.nodes, local_error(patch, me)


def windowed_align_errors(patches, output_file, window=14, scale=True, use_median=True):
    output_file = Path(output_file)
    if not output_file.is_file():
        n_nodes = num_nodes(patches).compute()
        n_patches = num_patches(patches).compute()
        workfile = output_file.with_suffix('.tmp.npy')
        try:
            errors = np.lib.format.open_memmap(workfile, mode='w+', dtype=float, shape=(n_nodes, n_patches))
            errors[:, :] = np.nan

            patch_errors = [last_patch_error(patches[i-window:i], scale, use_median).persist()
                            for i in range(window, n_patches+1)]
            secede()
            for i, err in zip(range(window-1, n_patches), patch_errors):
                nodes, patch_err = err.compute()
                errors[nodes, i] = patch_err
            rejoin()
            errors.flush()
        except Exception as e:
            workfile.unlink(missing_ok=True)
            raise e
        workfile.replace(output_file)
    return output_file
