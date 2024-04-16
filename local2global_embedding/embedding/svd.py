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


# modified from scipy.sparse.linalg.svds:
# Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def _augmented_orthonormal_cols(x, k):
    # extract the shape of the x array
    n, m = x.shape
    # create the expanded array and copy x into it
    y = np.empty((n, m+k), dtype=x.dtype)
    y[:, :m] = x
    # do some modified gram schmidt to add k random orthonormal vectors
    for i in range(k):
        # sample a random initial vector
        v = np.random.randn(n)
        if np.iscomplexobj(x):
            v = v + 1j*np.random.randn(n)
        # subtract projections onto the existing unit length vectors
        for j in range(m+i):
            u = y[:, j]
            v -= (np.dot(v, u.conj()) / np.dot(u, u.conj())) * u
        # normalize v
        v /= np.sqrt(np.dot(v, v.conj()))
        # add v into the output array
        y[:, m+i] = v
    # return the expanded array
    return y


def _augmented_orthonormal_rows(x, k):
    return _augmented_orthonormal_cols(x.T, k).T


def _svds_laplacian(A, k=6, tol=1e-8,
         maxiter=None, random_state=None, maxrestarts=10, verbose=0):
    """Compute the largest or smallest k singular values/vectors for a sparse matrix. The order of the singular values is not guaranteed.

    Parameters
    ----------
    A : {sparse matrix, LinearOperator}
        Array to compute the SVD on, of shape (M, N)
    k : int, optional
        Number of singular values and vectors to compute.
        Must be 1 <= k < min(A.shape).
    tol : float, optional
        Tolerance for singular values. Zero (default) means machine precision.

    maxiter : int, optional
        Maximum number of iterations.

        .. versionadded:: 0.12.0

    Returns
    -------
    u : ndarray, shape=(M, k)
        Unitary matrix having left singular vectors as columns.
        If `return_singular_vectors` is "vh", this variable is not computed,
        and None is returned instead.
    s : ndarray, shape=(k,)
        The singular values.
    vt : ndarray, shape=(k, N)
        Unitary matrix having right singular vectors as rows.



    Notes
    -----
    This is a naive implementation using  LOBPCG as an eigensolver
    on A.H * A or A * A.H, depending on which one is more efficient.
    """
    if maxiter is None:
        maxiter = max(k, 20)
    rg = np.random.default_rng(random_state)

    d1 = np.asarray(A.sum(axis=1)).flatten()
    D1 = ss.diags(d1 ** (-0.5))

    d2 = np.asarray(A.sum(axis=0)).flatten()
    D2 = ss.diags(d2 ** -0.5)
    A = D1 @ A @ D2
    n, m = A.shape

    if k <= 0 or k >= min(n, m):
        raise ValueError("k must be between 1 and min(A.shape), k=%d" % k)
    else:
        if n > m:
            X_dot = X_matmat = A.dot
            XH_dot = XH_mat = A.T.dot
            v0 = d2**0.5
        else:
            XH_dot = XH_mat = A.dot
            X_dot = X_matmat = A.T.dot
            v0 = d1**0.5

    def matvec_XH_X(x):
        return XH_dot(X_dot(x))

    def matmat_XH_X(x):
        return XH_mat(X_matmat(x))

    XH_X = sl.LinearOperator(matvec=matvec_XH_X, dtype=A.dtype,
                          matmat=matmat_XH_X,
                          shape=(min(A.shape), min(A.shape)))

    # Get a low rank approximation of the implicitly defined gramian matrix.
    # This is not a stable way to approach the problem.


    X = rg.normal(size=(min(A.shape), k))
    for _ in range(maxrestarts):
        eigvals, eigvec, res = sl.lobpcg(XH_X, X, Y=v0[:, None], tol=tol, maxiter=maxiter,
                                    largest=True, retResidualNormsHistory=True, verbosityLevel=verbose)
        if res[-1].max() > tol:
            X = eigvec + rg.normal(size=eigvec.shape, scale=0.5*tol)
        else:
            break

    # Gramian matrices have real non-negative eigenvalues.
    eigvals = np.maximum(eigvals.real, 0)

    # Use the sophisticated detection of small eigenvalues from pinvh.
    t = eigvec.dtype.char.lower()
    factor = {'f': 1E3, 'd': 1E6}
    cond = factor[t] * np.finfo(t).eps
    cutoff = cond * np.max(eigvals)

    # Get a mask indicating which eigenpairs are not degenerately tiny,
    # and create the re-ordered array of thresholded singular values.
    above_cutoff = (eigvals > cutoff)
    nlarge = above_cutoff.sum()
    nsmall = k - nlarge
    slarge = np.sqrt(eigvals[above_cutoff])
    s = np.zeros_like(eigvals)
    s[:nlarge] = slarge

    if n > m:
        vlarge = eigvec[:, above_cutoff]
        ularge = X_matmat(vlarge) / slarge
        vhlarge = vlarge.T
    else:
        ularge = eigvec[:, above_cutoff]
        vhlarge = (X_matmat(ularge) / slarge).T

    u = _augmented_orthonormal_cols(ularge, nsmall) if ularge is not None else None
    vh = _augmented_orthonormal_rows(vhlarge, nsmall) if vhlarge is not None else None

    indexes_sorted = np.argsort(s)
    s = s[indexes_sorted]
    if u is not None:
        u = u[:, indexes_sorted]
    if vh is not None:
        vh = vh[indexes_sorted]

    return D1 @ u, s, vh @ D2


def bipartite_svd_patches(A: ss.spmatrix, dim, verbose=0):
    """
    SVD embedding of bipartite network
    Args:
        A:
        dim:

    Returns:

    """
    index1, _ = A.sum(axis=1).nonzero()
    _, index2 = A.sum(axis=0).nonzero()
    R1 = ss.coo_matrix((np.ones(index1.size), (np.arange(index1.size), index1)), shape=(index1.size, A.shape[0]))
    R2 = ss.coo_matrix((np.ones(index2.size), (index2, np.arange(index2.size))), shape=(A.shape[1], index2.size))
    A = R1 @ A @ R2

    U, s, Vh = _svds_laplacian(A, k=dim, verbose=verbose)
    return Patch(index1, U), Patch(index2, Vh.T)

