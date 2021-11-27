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
from dask import bag

import numpy as np
from local2global.lazy import LazyMeanAggregatorCoordinates
from local2global_embedding.run.scripts.utils import mean_embedding, compute
from dask import delayed


def no_transform_embedding(patches, shape, output_file, mmap=True, use_tmp=True):
    print(f'launch no-transform embedding for {output_file} with {mmap=} and {use_tmp=}')
    output_file = Path(output_file)

    if mmap:
        mean_embedding(patches.compute(), shape, output_file, use_tmp)
    else:
        np.save(output_file, np.asarray(LazyMeanAggregatorCoordinates(patches.compute()), dtype=np.float32))
    return output_file
