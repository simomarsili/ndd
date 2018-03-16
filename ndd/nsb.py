# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)

__copyright__ = "Copyright (C) 2016,2017 Simone Marsili"
__license__ = "BSD 3 clause"
__author__ = "Simone Marsili (simomarsili@gmail.com)"
__all__ = ['entropy', 'histogram']

import numpy as np

def _check_histogram(counts, k=None, alpha=0.0):
    """Check that `counts` contains valid frequency counts."""

    try:
        # always flatten the input array
        counts = np.ravel(np.array(counts, dtype=np.int32))
    except ValueError:
        raise
    if np.any(counts < 0):
        raise ValueError("A bin cant have a frequency < 0")

    nbins = np.int32(len(counts))
    if k is None:
        k = nbins
    else:
        try:
            k = np.int32(k)
        except ValueError:
            raise
        if k < nbins:
            raise ValueError("k (%s) is smaller than the number of bins (%s)"
                             % (k, nbins))
    if alpha is None:
        alpha = 0.0
    else:
        try:
            alpha = np.float64(alpha)
        except ValueError:
            raise

    return (counts, k, alpha)

def entropy(counts, k=None, a=None, return_std=False, dist=False):
    """
    Return a Bayesian estimate of the entropy of an unknown discrete
    distribution from an input array of counts. The estimator relies on a
    mixture of (properly weighted) Dirichlet priors
    (Nemenman-Shafee-Bialek estimator).

    If `a` is passed, use a single Dirichlet prior with concentration
    parameter `a` (fixed alpha estimator).
    If `dist` == True, first estimate the underlying distribution over
    states/classes and then plug this estimate into the entropy definition
    (maximum likelihood estimator).
    If `a` is passed in combination with `dist=True`, the underlying
    distribution is approximated by adding `a` pseudocounts to the observed
    state frequencies (`pseudocount` estimator).

    Parameters
    ----------

    counts : array_like
        The number of occurrences of a set of states/classes.
        Non-1D arrays are interpreted as frequencies on a grid of bins
        (and flattened).

    k : int, optional
        Total number of classes. must be k >= len(counts).
        Defaults to len(counts).

    a : float, optional
        Concentration parameter of the Dirichlet prior.
        Must be >= 0.0. If no value is passed, use a mixture of Dirichlet
        priors (Nemenman-Schafee-Bialek algorithm).

    return_std : boolean, optional
        If True, also return the standard deviation over the posterior for H.

    dist : boolean, optional
        If True, the true underlying distribution is estimated from counts,
        and plugged in the entropy definition ("plugin" estimator).
        Use `a` as the concentration parameter for the Dirichlet prior
        ("pseudocount" estimator).
        If `a` is None, use the empirical distribution ("ML" estimatator).

    Returns
    -------
    entropy : float
        Entropy estimate.

    std : float, optional
        If return_std == True, return the standard deviation over the posterior
        for H. When dist == True, return None.

    """
    from ndd import _nsb

    counts, k, alpha = _check_histogram(counts, k, a)
    if k == 1: # if the total number of classes is one
        if return_std:
            return (0.0, 0.0)
        else:
            return 0.0

    std = None
    if dist:
        if alpha < 1e-6:
            # we'll take this as zero
            estimate = _nsb.plugin(counts)
        else:
            estimate = _nsb.pseudo(counts, k, alpha)
    else:
        if a is None:
            # NSB
            estimate, std = _nsb.nsb(counts, k)
        else:
            # fixed alpha
            estimate = _nsb.dirichlet(counts, k, alpha)
            #TODO: compute variance over the posterior at fixed alpha

    if estimate is np.nan:
        raise FloatingPointError("Estimate is NaN")

    if return_std:
        return (estimate, std)
    else:
        return estimate

def histogram(data, return_unique=False):
    """Compute an histogram from data. Wrapper to numpy.unique.

    Parameters
    ----------

    data : array_like
        Input data array. If n-dimensional, statistics is computed along
        axis 0.

    return_unique : bool, optional
        If True, also return the unique elements corresponding to each bin.

    Returns
    -------

    counts : ndarray
        Bin counts.

    unique : ndarray, optional
        Unique elements corresponding to each bin in counts.
        Only if return_elements == True

    """

    unique, counts = np.unique(data, axis=0, return_counts=True)

    if return_unique:
        return (unique, counts)
    else:
        return counts
