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
from dask.distributed import secede, rejoin

from local2global.utils import relative_scale, relative_orthogonal_transform
from local2global.patch import Patch
from local2global_embedding.run.scripts.utils import num_patches, num_nodes, get_dim


def temporal_align_errors(patches, output_file, scale=True):
    output_file = Path(output_file)
    if not output_file.is_file():
        secede()
        n_nodes = num_nodes(patches).compute()
        n_patches = num_patches(patches).compute()
        patch = patches[0].compute()
        rejoin()
        dim = patch.shape[1]
        reference = np.zeros((n_nodes, dim))
        counts = np.zeros((n_nodes), dtype=np.int32)
        counts[patch.nodes] = 1
        reference[patch.nodes] = patch.coordinates
        workfile = output_file.with_suffix('.tmp.npy')
        try:
            errors = np.lib.format.open_memmap(workfile, mode='w+', dtype=float, shape=(n_nodes, n_patches))
            errors[:, :] = np.nan
            errors[patch.nodes, 0] = 0.0

            for pi in range(1, n_patches):
                secede()
                patch = patches[pi].compute()
                rejoin()
                valid_nodes = patch.nodes[counts[patch.nodes] > 0]
                ref = reference[valid_nodes]
                coords = patch.get_coordinates(valid_nodes)
                if scale:
                    scale_factor = relative_scale(ref, coords)
                    patch.coordinates *= scale_factor
                rot = relative_orthogonal_transform(ref, coords)
                patch.coordinates = patch.coordinates @ rot.T
                patch.coordinates -= np.nanmean(coords, axis=0, keepdims=True)
                patch.coordinates += np.nanmean(ref, axis=0, keepdims=True)

                errors[valid_nodes, pi] = np.linalg.norm(ref - patch.get_coordinates(valid_nodes), axis=1)
                reference[patch.nodes] *= counts[patch.nodes, None]
                counts[patch.nodes] += 1
                reference[patch.nodes] += patch.coordinates
                reference[patch.nodes] /= counts[patch.nodes, None]

            errors.flush()
            workfile.replace(output_file)
        finally:
            workfile.unlink(missing_ok=True)
    return output_file




