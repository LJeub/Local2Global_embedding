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

from dask import delayed
from dask.distributed import secede, rejoin
import numpy as np

from local2global.utils import local_error
from local2global_embedding.patches import rolling_window_graph
from local2global_embedding.run.scripts.utils import aligned_coords, num_patches, num_nodes


def global_align_errors(patches, output_file, window=14, scale=True, verbose=False, use_tmp=False):
    output_file = Path(output_file)
    if not output_file.is_file():
        pg = delayed(rolling_window_graph)(num_patches(patches), window)
        secede()
        n_nodes = num_nodes(patches).compute()
        aligned = aligned_coords(patches, pg, verbose=verbose, scale=scale, use_tmp=use_tmp).compute()
        rejoin()

        workfile = output_file.with_suffix('.tmp.npy')
        ref_file = workfile.with_name(workfile.name.replace('error', 'reference'))
        try:
            errors = np.lib.format.open_memmap(workfile, mode='w+', dtype=float, shape=(n_nodes, len(aligned.patches)))
            reference = np.lib.format.open_memmap(ref_file, mode='w+', dtype=float, shape=(n_nodes, aligned.shape[1]))
            reference[:, :] = np.nan
            errors[:, :] = np.nan
            reference[aligned.nodes] = aligned.coordinates
            for i, p in enumerate(aligned.patches):
                errors[p.nodes, i] = local_error(p, reference)

            errors.flush()
            reference.flush()
            workfile.replace(output_file)
            ref_file.replace(ref_file.with_name(ref_file.name.replace('.tmp', '')))
        finally:
            workfile.unlink(missing_ok=True)
            ref_file.unlink(missing_ok=True)
    return output_file
