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

MAX_LOGK = 150 * numpy.log(2)  # 200 bits


def entropy(ar, k=None, alpha=None, return_std=False, plugin=False,
            counts='precomputed'):
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

    counts : 'precomputed' (default), None or int, optional
        By default ('precomputed') `ar` is an array of bin counts.
        If None or int, defines the axis indexing different samples in the data
        array `ar`. If None, compute counts on the flattened array.
        If int: compute counts along the given axis. In this case,
        the subarrays indexed by the axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the axis.

    Returns
    -------
    entropy : float
        Entropy estimate.

    std : float, optional
        Uncertainty in the entropy estimate
        (approximates the standard deviation over the entropy posterior).
        Only provided if `return_std` is True.

    """

    if counts == 'precomputed':
        try:
            freqs = numpy.array(ar, dtype=numpy.int32)
        except ValueError:
            raise
        if numpy.any(freqs < 0):
            raise ValueError("Frequency counts cant be negative")
        # flatten the input array; TODO: as numpy.unique
        freqs = freqs.flatten()
    else:
        # diffrent samples as different columns
        samples_axis = counts
        ar = ndd.nsb._2darray(ar, axis=samples_axis)
        ks = [len(numpy.unique(v)) for v in ar]
        freqs = ndd.histogram(ar, axis=1)

    n_bins = len(freqs)
    if k is None:
        if counts == 'precomputed':
            k = numpy.float64(n_bins)
        else:
            k = numpy.sum(numpy.log(x) for x in ks)
            if k > MAX_LOGK:
                # too large a number; backoff to n_bins?
                # TODO: log warning
                raise ValueError('k (%r) larger than %r' %
                                 (numpy.exp(k), numpy.exp(MAX_LOGK)))
            else:
                k = numpy.exp(k)
    else:
        try:
            k = numpy.float64(k)
        except ValueError:
            raise
        if numpy.log(k) > MAX_LOGK:
            raise ValueError('k (%r) larger than %r' %
                             (k, numpy.exp(MAX_LOGK)))
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
            result = ndd._nsb.plugin(freqs, k)
        else:
            result = ndd._nsb.pseudo(freqs, k, alpha)
    else:
        if alpha is None:
            result = ndd._nsb.nsb(freqs, k)
            if not return_std:
                result = result[0]
        else:
            # TODO: compute variance over the posterior at fixed alpha
            result = ndd._nsb.dirichlet(freqs, k, alpha)

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


def _2darray(ar, axis=0, to_axis=1):
    """
    For a 2D n-by-p data array, transpose it.
    For a generic ndarray, move axis `axis` to axis `to_axis`,
    and flatten the subarrays corresponding to other dimensions.
    """

    ar = numpy.asanyarray(ar)

    if ar.ndim == 1:
        n = ar.shape[0]
        ar = ar.reshape(n, 1)

    if axis != 0:
        try:
            ar = numpy.swapaxes(ar, axis, 0)
        except ValueError:
            raise numpy.AxisError(axis, ar.ndim)

    if ar.ndim > 2:
        n = ar.shape[0]
        ar = ar.reshape(n, -1)

    if to_axis == 1:
        ar = ar.T

    return numpy.ascontiguousarray(ar)


def _combinations(func, ar, ks=None, r=1):
    """
    Given a function and a n-by-p array of data, compute the function over all
    possible p-choose-r combinations of r columns.

    Paramaters
    ----------

    func : function
        Function taking as input a discrete data array and alphabet size:
        func(data, k=k).

    ar : array-like
        Array of n samples from p discrete variables.

    ks : 1D p-dimensional array, optional
        Alphabet size for each variable.
        The alphabet sizes for the r-dimensional variable corresponding to the
        column indices ix is computed as numpy.prod([k[x] for x in ix]).
        Defaults to the number of unique elements in each column.

    r : int, optional
        For each possible combination of r columns, return the estimated
        entropy for the corresponding r-dimensional variable.
        See itertools.combinations(range(p), r=r).
        Defaults to 1 (a different estimate for each column/variable).

    """
    from itertools import combinations

    ar = ndd.nsb._2darray(ar, axis=0)
    p, n = ar.shape

    try:
        if len(ks) == p:
            ks = numpy.array(ks)
        else:
            raise ValueError("k should have len %s" % p)
    except TypeError:
        if ks is None:
            ks = numpy.array([numpy.unique(v).size for v in ar])
        else:
            raise

    alphabet_sizes = (numpy.prod(x) for x in combinations(ks, r=r))
    data = combinations(ar, r=r)

    estimates = []
    for k, d in zip(alphabet_sizes, data):
        estimates.append(func(d, k=k))
    return estimates


def multivariate_information(a, ks=None):
    """docs."""
    import ndd

    try:
        n, p = a.shape
    except AttributeError:
        raise

    multi_info = 0.0
    for r in range(1, p+1):
        sgn = (-1)**r
        multi_info += sgn * numpy.sum(ndd._combinations(
            ndd.entropy, a, ks=ks, r=r))

    return - multi_info
