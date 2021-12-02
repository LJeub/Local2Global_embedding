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
import dask.array as da

from dask.distributed import wait, worker_client

from local2global_embedding.outliers import leave_out_nan_z_score
from local2global_embedding.run.scripts.utils import mmap_dask_array


def z_score_chunk(in_chunk, out_chunk):
    out_chunk[:] = leave_out_nan_z_score(in_chunk)
    return out_chunk


def leave_out_z_score_errors(error_file, blocksize=None):
    error_file = Path(error_file)
    output_file = error_file.with_name(error_file.name.replace('error', 'lo_z_score_error'))
    if not output_file.is_file():
        workfile = output_file.with_suffix('.tmp.npy')
        try:
            error_data = np.load(error_file, mmap_mode='r')
            if blocksize is None:
                shape = error_data.shape
                blocksize = int(min(2**24/shape[1], shape[0]))

            errors = mmap_dask_array(error_file, blocksize=blocksize)
            out = mmap_dask_array(workfile, shape=error_data.shape, dtype=error_data.dtype, blocksize=blocksize)
            out = da.map_blocks(z_score_chunk, errors, out, dtype=error_data.dtype, meta=np.array((), dtype=error_data.dtype))
            with worker_client() as client:
                task = client.compute(out)
                wait(task)
            workfile.replace(output_file)
        finally:
            workfile.unlink(missing_ok=True)
    return output_file
