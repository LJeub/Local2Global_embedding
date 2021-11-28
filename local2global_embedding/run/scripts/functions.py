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

"""
provides interface for accessing script functions
"""
import importlib


def __getattr__(name):
    if name == '__path__':
        raise AttributeError  # short circuit for __path__ attribute

    try:
        module = importlib.import_module('..' + name, __name__)
    except ModuleNotFoundError:
        raise AttributeError(f'No function named {name}')
    func = getattr(module, name)
    globals()[name] = func  # make future lookups fast
    return func

# from .evaluate import evaluate
# from .l2g_align_patches import l2g_align_patches
# from .prepare_patches import prepare_patches
# from .train import train
from .hierarchical_l2g_align_patches import hierarchical_l2g_align_patches
# from local2global_embedding.run.scripts.no_transform_embedding import no_transform_embedding
# from .utils import load_patches
# from .svd_patch import svd_patch
