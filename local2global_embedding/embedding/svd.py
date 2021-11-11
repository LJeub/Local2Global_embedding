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
import scipy.sparse as ss
import scipy.sparse.linalg as sl
import numpy as np

from local2global import Patch


def bipartite_svd_patches(A: ss.spmatrix, dim):
    """
    SVD embedding of bipartite network
    Args:
        A:
        dim:

    Returns:

    """
    d1 = np.asarray(A.sum(axis=1)).flatten()
    index1 = np.flatnonzero(d1 > 0)
    R1 = ss.coo_matrix((np.ones(index1.size), (np.arange(index1.size), index1)), shape=(index1.size, A.shape[0]))
    D1 = ss.diags(d1[index1]**(-0.5))

    d2 = np.asarray(A.sum(axis=0)).flatten()
    index2 = np.flatnonzero(d2 > 0)
    R2 = ss.coo_matrix((np.ones(index2.size), (index2, np.arange(index2.size))), shape=(A.shape[1], index2.size))
    D2 = ss.diags(d2[index2]**-0.5)
    A = D1 @ R1 @ A @ R2 @ D2
    U, s, Vh = sl.svds(A, k=dim)
    return Patch(index1, D1 @ U), Patch(index2, D2 @ Vh.T)

