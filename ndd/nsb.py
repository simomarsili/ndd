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


class Entropy(object):
    def __init__(self, alpha=None, plugin=False):
        self.std = None

        # check alpha value
        if alpha:
            try:
                alpha = numpy.float64(alpha)
            except ValueError:
                raise
            if alpha <= 0:
                raise ValueError("alpha <= 0")
        self.alpha = alpha

        # set estimator
        if plugin:
            if alpha is None:
                self.estimator = self._plugin
            else:
                self.estimator = lambda counts, k: self._pseudocounts(
                    counts, k, self.alpha)
        else:
            if alpha is None:
                self.estimator = self._nsb
            else:
                self.estimator = lambda counts, k: self._ww(
                    counts, k, self.alpha)

    @staticmethod
    def _plugin(counts, k):
        return ndd._nsb.plugin(counts, k)

    @staticmethod
    def _pseudocounts(counts, k, alpha):
        return ndd._nsb.pseudo(counts, k, alpha)

    @staticmethod
    def _ww(counts, k, alpha):
        return ndd._nsb.dirichlet(counts, k, alpha)

    @staticmethod
    def _nsb(counts, k):
        return ndd._nsb.nsb(counts, k)

    @staticmethod
    def check_counts(a):
        try:
            a = numpy.asarray(a, dtype=numpy.int32)
        except ValueError:
            raise
        if numpy.any(a < 0):
            raise ValueError("Frequency counts can't be negative")
        return a.flatten()

    @staticmethod
    def check_k(k, n_bins):
        """
        if k is None, set k = number of bins
        if k is an integer, just check
        """

        if k is None:
            # set k to the number of observed bins
            k = numpy.float64(n_bins)
        else:
            try:
                k = numpy.float64(k)
            except ValueError:
                raise
            # if a scalar check size
            if numpy.log(k) > MAX_LOGK:
                raise ValueError('k (%r) larger than %r' %
                                 (k, numpy.exp(MAX_LOGK)))
            # consistency checks
            if k < n_bins:
                raise ValueError("k (%s) is smaller than the number of bins"
                                 "(%s)" % (k, n_bins))
            if not k.is_integer():
                raise ValueError("k (%s) should be a whole number." % k)
        return k

    def __call__(self, counts, k=None, return_std=False):
        """
        counts : array_like
            The number of occurrences of a set of bins.

        k : int, optional
            Number of bins. k >= len(counts).
            Float is valid input for whole numbers (e.g. k=1.e3).
            Defaults to len(counts).

        """
        self.fit(counts, k)
        return self.entropy

    def fit(self, counts, k=None):
        """
        counts : array_like
            The number of occurrences of a set of bins.

        k : int, optional
            Number of bins. k >= len(counts).
            Float values are valid input for whole numbers (e.g. k=1.e3).
            Defaults to len(counts).

        """
        counts = self.check_counts(counts)
        k = self.check_k(k=k, n_bins=len(counts))
        if k == 1:  # single bin
            self.entropy = self.std = 0.0
        else:
            result = self.estimator(counts, k)
            if isinstance(result, tuple):
                self.entropy, self.std = result
            else:
                self.entropy = result


def entropy(counts, k=None, alpha=None, return_std=False, plugin=False):
    """
    Return a Bayesian estimate of the entropy of an unknown discrete
    distribution from an input array of counts.

    Parameters
    ----------

    counts : array_like
        The number of occurrences of a set of bins.

    k : int, optional
        Number of bins. k >= len(counts).
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
        distribution is estimated from the empirical frequencies over bins
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

    estimator = Entropy(alpha, plugin)
    estimator.fit(counts, k)

    if return_std:
        result = estimator.entropy, estimator.std
    else:
        result = estimator.entropy

    if numpy.any(numpy.isnan(numpy.squeeze(result))):
        raise FloatingPointError("NaN value")

    return result


def data_entropy(data, k=None, alpha=None, return_std=False, plugin=False):
    """
    Return a Bayesian estimate of the entropy of an unknown discrete
    distribution from data.

    Parameters
    ----------

    data : array-like or generator
        If a generator, data must return hashable objects (e.g. tuples) as
        1D samples. Otherwise, data will be treated as an array of n samples
        from p variables.

    k : int, optional
        Number of bins. Floats are valid input for whole numbers (e.g. k=1.e3).
        Defaults to the number of unique objects in data (if 1D),
        or to the product of the number of unique elements for each variable.

    alpha : float, optional
        If alpha is passed, use a single Dirichlet prior with concentration
        parameter alpha (fixed alpha estimator). alpha > 0.0.

    return_std : boolean, optional
        If True, also return an approximated value for the standard deviation
        over the entropy posterior.

    plugin : boolean, optional
        If True, return a 'plugin' estimate of the entropy. The discrete
        distribution is estimated from the empirical frequencies over bins
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

    counts, k1 = ndd.histogram(data)
    if k is None:
        k = k1

    estimator = Entropy(alpha, plugin)
    estimator.fit(counts, k)

    if return_std:
        result = estimator.entropy, estimator.std
    else:
        result = estimator.entropy

    if numpy.any(numpy.isnan(numpy.squeeze(result))):
        raise FloatingPointError("NaN value")

    return result


def histogram(data):
    """Compute an histogram from data. Wrapper to numpy.unique.

    Parameters
    ----------

    data : array-like or generator
        If a generator, data must return hashable objects (e.g. tuples) as
        1D samples. Otherwise, data will be treated as an array of n samples
        from p variables.

    Returns
    -------

    counts : ndarray
        Bin counts.

    k : int
        The number of unique elements along axis 0. If data is p-dimensional,
        the product of the number of unique elements of each variable.

    """
    import inspect
    if inspect.isgenerator(data):
        from collections import Counter
        counter = Counter(data)
        counts = list(counter.values())
        k = len(counts)
    else:
        # reshape as a p-by-n array
        data = ndd.nsb._2darray(data)
        # number of unique elements for each of the p variables
        ks = [len(numpy.unique(v)) for v in data]
        logk = numpy.sum(numpy.log(x) for x in ks)
        if logk > MAX_LOGK:
            # too large a number; backoff to n_bins?
            # TODO: log warning
            raise ValueError('k (%r) larger than %r' %
                             (numpy.exp(logk), numpy.exp(MAX_LOGK)))
        else:
            k = numpy.prod(ks)

        # statistics for the p-dimensional variable
        _, counts = numpy.unique(data, return_counts=True, axis=1)
    return counts, k


def _2darray(ar):
    """
    For a 2D n-by-p data array, transpose it.
    For a generic ndarray, flatten the subarrays indexed by axis 0
    """

    ar = numpy.asanyarray(ar)

    if ar.ndim == 1:
        n = ar.shape[0]
        ar = ar.reshape(n, 1)

    if ar.ndim > 2:
        n = ar.shape[0]
        ar = ar.reshape(n, -1)

    ar = ar.T

    return numpy.ascontiguousarray(ar)


def _combinations(f, ar, ks=None, r=1):
    """
    Given an estimator `f` and a n-by-p array of data, apply f over all
    possible p-choose-r combinations of r columns.

    Paramaters
    ----------

    f : estimator
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

    ar = _2darray(ar)
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
        estimates.append(f(d, k=k))
    return estimates
