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

from dask.distributed import worker_client, as_completed

from local2global_embedding.outliers import leave_out_nan_z_score
from local2global_embedding.run.utils import watch_progress


@delayed
def z_score_chunk(input_file, output_file, chunk):
    in_chunk = np.load(input_file, mmap_mode='r')[chunk]
    out_chunk = np.load(output_file, mmap_mode='r+')[chunk]
    out_chunk[:] = leave_out_nan_z_score(in_chunk)
    out_chunk.flush()


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
            out_data = np.lib.format.open_memmap(workfile, mode='w+', shape=error_data.shape, dtype=error_data.dtype)
            out_data.flush()
            tasks = []
            for i in range(0, error_data.shape[0], blocksize):
                tasks.append(z_score_chunk(error_file, workfile, slice(i, i+blocksize)))
            with worker_client() as client:
                tasks = client.compute(tasks)
                tasks = as_completed(tasks)
                watch_progress(tasks)

            workfile.replace(output_file)
        finally:
            workfile.unlink(missing_ok=True)
    return output_file
