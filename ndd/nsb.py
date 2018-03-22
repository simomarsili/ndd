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

import numpy

def _check_counts(counts):
    """Check that `counts` contains valid frequency counts."""

    try:
        counts = numpy.array(counts, dtype=numpy.int32)
    except ValueError:
        raise

    if numpy.any(counts < 0):
        raise ValueError("Frequency counts cant be negative")

    # flatten the input array
    return counts.flatten()

def _check_k(n_bins, k=None):
    """Check the total number of classes `k`."""

    if k is None:
        k = numpy.int32(n_bins)
    else:
        try:
            k = numpy.int32(k)
        except ValueError:
            raise
        if k < n_bins:
            raise ValueError("k (%s) is smaller than the number of bins (%s)"
                             % (k, n_bins))
    return k

def _check_alpha(alpha):
    """Check the value of the concentration parameter."""

    if alpha:
        try:
            alpha = numpy.float64(alpha)
        except ValueError:
            raise
        if alpha <= 0:
            raise ValueError("alpha <= 0")

    return alpha

def _set_estimator(k, alpha, dist):
    from ndd import _nsb

    args_dict = {
        _nsb.plugin: (),
        _nsb.pseudo: (k, alpha),
        _nsb.nsb: (k, ),
        _nsb.dirichlet: (k, alpha)
    }

    if dist:
        if alpha is None:
            estimator = _nsb.plugin
        else:
            estimator = _nsb.pseudo
    else:
        if alpha is None:
            estimator = _nsb.nsb
        else:
            # fixed alpha
            estimator = _nsb.dirichlet
            #TODO: compute variance over the posterior at fixed alpha

    return (estimator, args_dict[estimator])

def entropy(counts, k=None, alpha=None, return_std=False, dist=False):
    """
    Return a Bayesian estimate of the entropy of an unknown discrete
    distribution from an input array of counts. The estimator relies on a
    mixture of (properly weighted) Dirichlet priors
    (Nemenman-Shafee-Bialek estimator).

    Parameters
    ----------

    counts : array_like
        The number of occurrences of a set of states/classes.
        The estimate is computed over the flattened array.

    k : int, optional
        Total number of classes. must be k >= len(counts).
        Defaults to len(counts).

    alpha : float, optional
        If `alpha` is passed, use a single Dirichlet prior with concentration
        parameter alpha (fixed alpha estimator). Must be > 0.0.

    return_std : boolean, optional
        If True, also return the standard deviation over the posterior for H.

    dist : boolean, optional
        If True, first estimate the underlying distribution over
        states/classes and then plug this estimate into the entropy definition
        (plugin estimator).
        If `alpha` is passed in combination with `dist=True`, the underlying
        distribution is approximated by adding alpha pseudocounts to
        the observed frequencies (pseudocount estimator).

    Returns
    -------
    entropy : float
        Entropy estimate.

    std : float, optional
        If return_std == True, return an approximation for the standard
        deviation over the posterior for the entropy.

    """
    from ndd import _nsb

    counts = _check_counts(counts)

    k = _check_k(len(counts), k)

    alpha = _check_alpha(a)

    estimator, args = _set_estimator(k, alpha, dist)

    if k == 1: # if the total number of classes is one
        if estimator == _nsb.nsb and return_std:
            return (0.0, 0.0)
        else:
            return 0.0

    result = estimator(counts, *args)

    if numpy.any(numpy.isnan(numpy.squeeze(result))):
        raise FloatingPointError("NaN value")

    if estimator == _nsb.nsb:
        if not return_std:
            result = result[0]

    return result

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

    unique, counts = numpy.unique(data, axis=0, return_counts=True)

    if return_unique:
        return (unique, counts)
    else:
        return counts
