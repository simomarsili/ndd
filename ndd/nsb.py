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
import ndd
import ndd._nsb


def entropy(ar, k=None, alpha=None, return_std=False, plugin=False,
            axis='precomputed'):
    """
    Return a Bayesian estimate of the entropy of an unknown discrete
    distribution from an input array of counts. The estimator uses a mixture of
    Dirichlet priors (Nemenman-Shafee-Bialek estimator), with weights chosen
    such that the induced prior over the entropy is approximately uniform.

    Parameters
    ----------

    ar : array_like
        The number of occurrences of a set of states/classes
        (or an array of data samples, see the `axis` keyword arg).

    k : int, optional
        Total number of classes. k >= len(counts).
        A float value is a valid input for whole numbers (e.g. k=1.e3).
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

    axis : 'precomputed' (default), None or int, optional
        By default ('precomputed') `ar` is an array of precomputed counts.
        If None or int, counts are computed from `ar`, that is taken as
        an ndarray of data, and `axis` defines the axis indexing different
        samples (see also numpy.unique docstrings, axis kw).
        If None, frequencies are computed on the flattened array.
        If int: frequencies are computed along axis `axis`. The subarrays
        indexed by the given axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the given axis.

    Returns
    -------
    entropy : float
        Entropy estimate.

    std : float, optional
        Uncertainty in the entropy estimate
        (approximates the standard deviation over the entropy posterior).
        Only provided if `return_std` is True.

    """

    if axis == 'precomputed':
        try:
            counts = numpy.array(ar, dtype=numpy.int32)
        except ValueError:
            raise
        if numpy.any(counts < 0):
            raise ValueError("Frequency counts cant be negative")
        # flatten the input array; TODO: as numpy.unique
        counts = counts.flatten()
    else:
        # difefrent samples in different columns
        ar = ndd.nsb._2D_array(ar, axis=axis, transpose=True)
        ks = [len(numpy.unique(v)) for v in ar]
        counts = ndd.histogram(ar, axis=1)

    n_bins = len(counts)
    if k is None:
        if axis == 'precomputed':
            k = numpy.float64(n_bins)
        else:
            k = numpy.prod(ks)
    else:
        try:
            k = numpy.float64(k)
        except ValueError:
            raise
        if k < n_bins:
            raise ValueError("k (%s) is smaller than the number of bins (%s)"
                             % (k, n_bins))
        if not k.is_integer():
            raise ValueError("k (%s) should be a whole number.")

    if k == 1:  # if the total number of classes is one
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
            result = ndd._nsb.plugin(counts, k)
        else:
            result = ndd._nsb.pseudo(counts, k, alpha)
    else:
        if alpha is None:
            result = ndd._nsb.nsb(counts, k)
            if not return_std:
                result = result[0]
        else:
            # TODO: compute variance over the posterior at fixed alpha
            result = ndd._nsb.dirichlet(counts, k, alpha)

    if numpy.any(numpy.isnan(numpy.squeeze(result))):
        raise FloatingPointError("NaN value")

    return result


def histogram(data, return_unique=False, axis=None):
    """Compute an histogram from data. Wrapper to numpy.unique.

    Parameters
    ----------

    data : array_like
        Input data array.
        If n-dimensional, statistics is computed over
        axis 0.


    return_unique : bool, optional
        If True, also return the unique elements corresponding to each bin.

    axis : int, optional
        The axis to operate on. If None, `counts` will be flattened.
        If an integer, the subarrays indexed by the given axis will be
        flattened and treated as the elements of a 1-D array with the dimension
        of the given axis. Object arrays or structured arrays
        that contain objects are not supported if the `axis` kwarg is used. The
        default is None.
        Check numpy.unique docstrings for more details.

    Returns
    -------

    counts : ndarray
        Bin counts.

    unique : ndarray, optional
        Unique elements corresponding to each bin in counts.
        Only if return_elements == True

    """

    unique, counts = numpy.unique(data, return_counts=True, axis=axis)

    if return_unique:
        return (unique, counts)
    else:
        return counts


def _2D_array(ar, axis=0, transpose=False):
    """Transform a generic N-dimensional ndarray to a 2-D array.
       (see numpy.unique)
    """

    ar = numpy.asanyarray(ar)
    if axis != 0:
        try:
            ar = numpy.swapaxes(ar, axis, 0)
        except numpy.AxisError:
            raise numpy.AxisError(axis, ar.ndim)

    if ar.ndim > 2:
        n = ar.shape[0]
        ar = ar.reshape(n, -1)

    if transpose:
        ar = ar.T

    return numpy.ascontiguousarray(ar)


def entropy_combinations(ar, k=None, r=1, axis=0):
    """n-by-p data matrix"""
    from itertools import combinations

    # put samples on different columns
    ar = ndd.nsb._2D_array(ar, transpose=True)
    p, n = ar.shape

    try:
        if len(k) == p:
            k = numpy.array(k)
        else:
            raise ValueError("k should have len %s" % p)
    except TypeError:
        if k is None:
            k = numpy.array([numpy.unique(v).size for v in ar])
        else:
            raise

    estimates = []
    for ix in combinations(range(p), r=r):
        ix = list(ix)
        alphabet_size = numpy.prod(k[ix])
        estimates.append(ndd.entropy(ar[ix], k=alphabet_size, axis=1))
    return estimates


def combinations(func, a, k=None, r=1):
    """
    Evaluate function func(a, k) over all possible combinations of r variables
    from a set of p variables.

    Parameters
    ----------

    a : 2D array-like
        n-by-p matrix containing n samples of p discrete variables.

    r : int, optional
        For each possible combination of r columns, return the estimated
        entropy for the corresponding r-dimensional variable.
        See itertools.combinations(range(n), r=r).
        Defaults to 1 (a different estimate for each column/variable).

    k : 1D p-dimensional array, optional
        The alphabet size for the r-dimensional variable corresponding to the
        column indices ix is computed as numpy.prod([k[x] for x in ix]).
        Defaults to the product of the number of unique elements
        in each column.

    Returns
    -------
    entropy : list
        Entropy estimates.

    """
    import itertools

    try:
        n, p = a.shape
        at = a.T
    except AttributeError:
        raise

    if r <= 0:
        raise ValueError("r must be > 0 (input value: %s)" % r)

    try:
        if len(k) == p:
            ks = list(k)
        else:
            raise ValueError("k should have len %s" % p)
    except TypeError:
        if k is None:
            ks = [numpy.unique(v).size for v in at]
        else:
            raise
    ks = numpy.array(ks)

    estimates = []
    for ix in itertools.combinations(range(p), r=r):
        ix = list(ix)
        # at is transposed, samples are on different columns
        estimates.append(func(at[ix], k=numpy.prod(ks[ix]), axis=1))
    return estimates


def multivariate_information(a, k=None):
    """docs."""
    import ndd

    try:
        n, p = a.shape
        at = a.T
    except AttributeError:
        raise

    multi_info = 0.0
    for r in range(1, p+1):
        sgn = (-1)**r
        multi_info += sgn * numpy.sum(ndd.combinations(ndd.entropy_fromsamples, a, k=k, r=r))

    return - multi_info
