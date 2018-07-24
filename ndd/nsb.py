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

def entropy(counts, k=None, alpha=None, return_std=False, plugin=False):
    """
    Return a Bayesian estimate of the entropy of an unknown discrete
    distribution from an input array of counts. The estimator uses a mixture of
    Dirichlet priors (Nemenman-Shafee-Bialek estimator), with weights chosen
    such that the induced prior over the entropy is approximately uniform.

    Parameters
    ----------

    counts : array_like
        The number of occurrences of a set of states/classes.
        The entropy estimate is computed over the flattened array.

    k : int, optional
        Total number of classes. k >= len(counts).
        Defaults to len(counts).

    alpha : float, optional
        If alpha is passed, use a single Dirichlet prior with concentration
        parameter alpha (fixed alpha estimator). alpha > 0.0.

    return_std : boolean, optional
        If True, also return an approximated value for the standard deviation
        over the entropy posterior.

    plugin : boolean, optional
        If True, return a 'plugin' estimate of the entropy. The discrete
        distribution is estimated from the empirical frequencies over states
        and inserted into the entropy definition (plugin estimator).
        If alpha is passed in combination with plugin=True, add
        alpha pseudocounts to each frequency count (pseudocount estimator).

    Returns
    -------
    entropy : float
        Entropy estimate.

    std : float, optional
        Uncertainty in the entropy estimate
        (approximates the standard deviation over the entropy posterior).
        Only provided if `return_std` is True.

    """
    from ndd import _nsb

    try:
        counts = numpy.array(counts, dtype=numpy.int32)
    except ValueError:
        raise
    if numpy.any(counts < 0):
        raise ValueError("Frequency counts cant be negative")
    # flatten the input array
    counts = counts.flatten()

    n_bins = len(counts)
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

    if k == 1: # if the total number of classes is one
        if return_std:
            return (0.0, 0.0)
        else:
            return 0.0

    if alpha:
        try:
            alpha = numpy.float64(alpha)
        except ValueError:
            raise
        if alpha <= 0:
            raise ValueError("alpha <= 0")

    if plugin:
        if alpha is None:
            result = _nsb.plugin(counts)
        else:
            result =  _nsb.pseudo(counts, k, alpha)
    else:
        if alpha is None:
            result = _nsb.nsb(counts, k)
            if not return_std:
                result = result[0]
        else:
            #TODO: compute variance over the posterior at fixed alpha
            result = _nsb.dirichlet(counts, k, alpha)

    if numpy.any(numpy.isnan(numpy.squeeze(result))):
        raise FloatingPointError("NaN value")

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
