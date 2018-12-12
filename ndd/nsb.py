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

import functools
import numpy


def entropy(counts, k=None, alpha=None, return_std=False, plugin=False,
            axis=None):
    """
    Return a Bayesian estimate of the entropy of an unknown discrete
    distribution from an input array of counts. The estimator uses a mixture of
    Dirichlet priors (Nemenman-Shafee-Bialek estimator), with weights chosen
    such that the induced prior over the entropy is approximately uniform.

    Parameters
    ----------

    counts : array_like
        The number of occurrences of a set of states/classes.
        It will be flattened if it is not already 1-D.

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
        k = numpy.float64(n_bins)
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
            result = _nsb.plugin(counts, k)
        else:
            result = _nsb.pseudo(counts, k, alpha)
    else:
        if alpha is None:
            result = _nsb.nsb(counts, k)
            if not return_std:
                result = result[0]
        else:
            # TODO: compute variance over the posterior at fixed alpha
            result = _nsb.dirichlet(counts, k, alpha)

    if numpy.any(numpy.isnan(numpy.squeeze(result))):
        raise FloatingPointError("NaN value")

    return result


def histogram(data, return_unique=False):
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

    unique, counts = numpy.unique(data, return_counts=True, axis=0)

    if return_unique:
        return (unique, counts)
    else:
        return counts


def entropy_fromsamples(a, k=None, axis=0):
    """
    NSB entropy estimate from data samples.

    Parameters
    ----------

    a : 2D array-like
        n-by-p matrix containing n samples of p discrete variables.
        (see axis keyword).

    k : int or 1D p-dimensional array or int, optional
        If int, the number of possible outcomes, or alphabet size.
        If k is an array, the alphabet size is computed as numpy.prod(k).
        Defaults to the product of the number of unique elements
        observed for each variable.

    axis : int, optional
        If 0 (default), the different rows of the n-by-p matrix a
        correspond to different different samples.
        If 1, different columns correspond to different samples and
        the data matrix is p-by-n.

    Returns
    -------
    entropy : float
        Entropy estimate.

    """
    import ndd

    try:
        if axis == 0:
            n, p = a.shape
            a = a.T
        elif axis == 1:
            p, n = a.shape
        else:
            raise ValueError("axis value must be either 0 or 1")
    except AttributeError:
        raise

    try:
        if len(k) == p:
            ks = list(k)
            alphabet_size = numpy.prod(ks)
        else:
            raise ValueError("k should have len %s" % p)
    except TypeError:
        if k is None:
            ks = [numpy.unique(v).size for v in a]
            alphabet_size = numpy.prod(ks)
        else:
            alphabet_size = k

    data = list(zip(*a))
    counts = ndd.histogram(data)
    return ndd.entropy(counts, k=alphabet_size)


# this should be a decorator
def combinations(func, a, k=None, r=1):
    """
    NSB entropy estimates from data samples.

    Parameters
    ----------

    a : 2D array-like
        n-by-p matrix containing m samples of p discrete variables.

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
        # at is transposed, samples on different columns
        estimates.append(func(at[ix], ks[ix], axis=1))
    return estimates
