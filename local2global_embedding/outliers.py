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


import numpy as np
from collections.abc import Iterable

from tqdm.auto import tqdm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from local2global import AlignmentProblem
from local2global.utils import local_error
from local2global.patch import MeanAggregatorPatch


def all_errors(prob: AlignmentProblem):
    """
    Compute un-normalised alignment errors for all patches

    Args:
        prob: Alignment problem

    Returns: errors

    """
    errors = np.full((prob.n_nodes, prob.n_patches), float('nan'))
    reference = prob.mean_embedding()
    for i, p in enumerate(prob.patches):
        errors[p.nodes, i] = local_error(p, reference)
    return errors


def sliding_window_errors(prob: AlignmentProblem, window=7):
    errors = np.full((prob.n_nodes, prob.n_patches-window+1), float('nan'))
    for i in range(prob.n_patches-window+1):
        p = prob.patches[i+window-1]
        reference = np.full((prob.n_nodes, prob.dim), float('nan'))
        agg = MeanAggregatorPatch(prob.patches[i:i+window])
        reference[agg.nodes] = agg.coordinates
        errors[p.nodes, i] = local_error(p, reference)
    return errors


def nan_z_score(errors, axis=1):
    m_errors = np.nanmean(errors, axis=axis, keepdims=True)
    std_errors = np.nanstd(errors, axis=axis, keepdims=True)
    return (errors-m_errors) / (std_errors + 1e-16)


def bootstrap_nan_z_score(errors, repeats=100, random_state=None):
    rg = np.random.default_rng(random_state)
    samples = rg.integers(errors.shape[1], size=(errors.shape[1], repeats))
    b_errors = errors[:, samples]
    m_errors = np.nanmedian(np.nanmean(b_errors, axis=1, keepdims=True), axis=2)
    std_errors = np.nanmedian(np.nanstd(b_errors, axis=1, keepdims=True), axis=2)
    return (errors - m_errors) / (std_errors + 1e-16)


def leave_out_nan_z_score(errors):
    z_errors = np.full(errors.shape, float('nan'))
    for i in range(errors.shape[1]):
        mask = np.ones(errors.shape[1], dtype=bool)
        mask[i] = False
        m_e = np.nanmean(errors[:, mask], axis=1)
        s_e = np.nanstd(errors[:, mask], axis=1)
        z_errors[:, i] = (errors[:, i] - m_e) / (s_e + 1e-16)
    return z_errors


def leave_out_nan_rms_score(errors):
    z_errors = np.full(errors.shape, float('nan'))
    for i in range(errors.shape[1]):
        mask = np.ones(errors.shape[1], dtype=bool)
        mask[i] = False
        m_e = np.sqrt(np.nanmean((errors[:, mask]**2), axis=1))
        z_errors[:, i] = (errors[:, i]) / (m_e + 1e-16)
    return z_errors


def z_score_errors(prob: AlignmentProblem):
    """
    Compute z-score alignment errors using centroid over all patches as reference

    Args:
        prob: AlignmentProblem

    Returns: array of z-score errors

    Notes
        This function does not align the patches before computing the reference.

    """

    errors = all_errors(prob)
    return nan_z_score(errors, axis=1)


def leave_out_z_score_errors(prob: AlignmentProblem):
    errors = all_errors(prob)
    return leave_out_nan_z_score(errors)


def z_score_boost_errors(prob: AlignmentProblem, threshold=4, positive_only=True):
    """
    Boosted z-score errors

    Args:
        prob: Alignment problem
        threshold: boosting threshold

    Returns: boosted z-score errors

    This function first computes normal z-scores. It then uses `threshold` to identify outliers.
    If `positive_only=True` (the default), only positive z-scores are considered as outliers. The identified outliers
    are removed when computing the means and standard deviations and the z-scores are recomputed based on the
    outlier-corrected mean and standard deviation.

    """
    errors = all_errors(prob)
    z_errors = nan_z_score(errors, axis=1)
    if positive_only:
        outliers = z_errors > threshold
    else:
        outliers = np.abs(z_errors) > threshold
    in_errors = errors.copy()
    in_errors[outliers] = float('nan')
    m_errors = np.nanmean(in_errors, axis=1, keepdims=True)
    std_errors = np.nanstd(in_errors, axis=1, keepdims=True)
    return (errors-m_errors) / (std_errors + 1e-16)


def quantile_errors(errors, quantile=0.75):
    """
    errors scaled by provided quantile of the distribution

    """
    if isinstance(quantile, Iterable):
        q = np.nanquantile(errors, max(quantile), axis=1, keepdims=True) - np.nanquantile(errors, min(quantile), axis=1, keepdims=True)
    else:
        q = np.nanquantile(errors, quantile, axis=1, keepdims=True)
    m = np.nanmedian(errors, axis=1, keepdims=True)
    return (errors-m) / q


def LOF_error(prob, min_points=21):
    if isinstance(min_points, Iterable):
        lof = [LocalOutlierFactor(n_neighbors=n) for n in min_points]
        min_points = max(min_points)
    else:
        lof = [LocalOutlierFactor(n_neighbors=min_points)]
    out = np.full((prob.n_nodes, prob.n_patches), np.nan)
    for i, pids in tqdm(enumerate(prob.patch_index), total=prob.n_nodes):
        if len(pids) > min_points:
            points = np.array([prob.patches[pid].get_coordinate(i) for pid in pids])
            out[i, pids] = -1 - np.min([l.fit(points).negative_outlier_factor_ for l in lof], axis=0)
    return out


def iForest_error(prob):
    out = np.full((prob.n_nodes, prob.n_patches), np.nan)
    iforest = IsolationForest()
    for i, pids in tqdm(enumerate(prob.patch_index), total=prob.n_nodes):
        if pids:
            points = np.array([prob.patches[pid].get_coordinate(i) for pid in pids])
            iforest.fit(points)
            out[i, pids] = -iforest.score_samples(points)
    return out


def plt_score(score, active, score_range=None):
    import matplotlib.pyplot as plt

    if score_range is not None:
        if score_range.shape[0] == 4:
            err = score_range[1:3]
            plt.plot(score_range[0], 'k^', markersize=1)
            plt.plot(score_range[-1], 'kv', markersize=1)
        else:
            err = score_range
        err[0] = score - err[0]
        err[1] -= score
        plt.errorbar(np.arange(len(score)), score, fmt='none', yerr=err, linewidth=0.1, capsize=2, capthick=0.1, ecolor='k')
    plt.plot(score, '.-k', linewidth=0.1)
    plt.plot(active, score[active], '.r', label='red-team active')
    plt.legend()