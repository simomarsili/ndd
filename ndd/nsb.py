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
__all__ = ['entropy', 'histogram', 'from_data', 'nbins']

import numpy
import ndd
import ndd._nsb


class BaseEstimator(object):
    def __init__(self):
        self.estimate = None
        self.std = None

    def _check(self):
        # check input data
        raise NotImplementedError

    def fit(self):
        # set self.estimate, self.std
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Return estimate from input data. Delegate to fit."""
        self.fit(*args, **kwargs)
        return self.estimate


class Entropy(BaseEstimator):
    def __init__(self, alpha=None, plugin=False):
        super().__init__()

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

    def _check(self, counts, k):
        counts = self._check_counts(a=counts)
        k = self._check_k(k=k, n_bins=len(counts))
        return counts, k

    @staticmethod
    def _check_counts(a):
        try:
            a = numpy.asarray(a, dtype=numpy.int32)
        except ValueError:
            raise
        if numpy.any(a < 0):
            raise ValueError("Frequency counts can't be negative")
        return a.flatten()

    @staticmethod
    def _check_k(k, n_bins):
        """
        if k is None, set k = number of bins
        if k is an integer, just check
        ik an array set k = prod(k)
        """
        MAX_LOGK = 150 * numpy.log(2)

        if k is None:
            # set k to the number of observed bins
            k = numpy.float64(n_bins)
        else:
            try:
                k = numpy.float64(k)
            except ValueError:
                raise
            if k.ndim:
                # if k is a sequence, set k = prod(k)
                if k.ndim > 1:
                    raise ValueError('k must be a scalar or 1D array')
                logk = numpy.sum(numpy.log(x) for x in k)
                if logk > MAX_LOGK:
                    # too large a number; backoff to n_bins?
                    # TODO: log warning
                    raise ValueError('k (%r) larger than %r' %
                                     (numpy.exp(logk), numpy.exp(MAX_LOGK)))
                else:
                    k = numpy.prod(k)
            else:
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

    def fit(self, counts, k=None):
        """
        counts : array_like
            The number of occurrences of a set of bins.

        k : int, optional
            Number of bins. k >= len(counts).
            Float values are valid input for whole numbers (e.g. k=1.e3).
            Defaults to len(counts).

        """
        counts, k = self._check(counts, k)
        if k == 1:  # single bin
            self.estimate = self.std = 0.0
        else:
            result = self.estimator(counts, k)
            if isinstance(result, tuple):
                self.estimate, self.std = result
            else:
                self.estimate = result


def entropy(counts, k=None, alpha=None, return_std=False, plugin=False):
    """
    Return a Bayesian estimate of the entropy of an unknown discrete
    distribution from an input array of counts.

    Parameters
    ----------

    counts : array_like
        The number of occurrences of a set of bins.

    k : int or array-like, optional
        Number of bins. k >= len(counts).
        A float value is a valid input for whole numbers (e.g. k=1.e3).
        If an array, set k = numpy.prod(k).
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
        result = estimator.estimate, estimator.std
    else:
        result = estimator.estimate

    if numpy.any(numpy.isnan(numpy.squeeze(result))):
        raise FloatingPointError("NaN value")

    return result


def nbins(data):
    """
    The number of unique elements along axis 0. If data is p-dimensional,
    the num. of unique elements for each variable.
    """
    # reshape as a p-by-n array
    data = ndd.nsb._as_data_array(data)
    return [len(numpy.unique(v)) for v in data]


def histogram(data, axis=0, r=0):
    """Compute an histogram from data. Wrapper to numpy.unique.

    Parameters
    ----------
    data : array-like
        An array of n samples from p variables.
    axis : int, optional
        The sample-indexing axis
    r : int, optional
        If r > 0, return a generator that yields bin counts for each possible
        combination of r variables.

    Returns
    -------
    counts : ndarray
        Bin counts.

    """
    from itertools import combinations
    # reshape as a p-by-n array
    data = ndd.nsb._as_data_array(data, axis=axis)
    p, n = data.shape
    if r > p:
        raise ValueError('r (%r) is larger than the number of variables (%r)'
                         % (r, p))
    if r == 0:
        # statistics for the p-dimensional variable
        _, counts = numpy.unique(data, return_counts=True, axis=1)
        return counts
    else:
        return (ndd.histogram(d, axis=1) for d in combinations(data, r=r))


def _as_data_array(ar, axis=0):
    """
    For a 2D n-by-p data array, transpose it.
    For a generic ndarray, flatten the subarrays indexed by axis 0
    axis : int, optional
        The sample-indexing axis
    """

    ar = numpy.asanyarray(ar)

    if ar.ndim == 1:
        n = ar.shape[0]
        ar = ar.reshape(1, n)
    elif ar.ndim == 2:
        if axis == 0:
            ar = ar.T
    elif ar.ndim > 2:
        if axis != 0:
            try:
                ar = numpy.swapaxes(ar, axis, 0)
            except ValueError:
                raise numpy.AxisError(axis, ar.ndim)
        n = ar.shape[0]
        ar = ar.reshape(n, -1)
        ar = ar.T

    return numpy.ascontiguousarray(ar)


def from_data(ar, ks=None, axis=0, r=0):
    """
    Given an array of data, return an entropy estimate.

    Paramaters
    ----------
    ar : array-like
        n-by-p array of n samples from p discrete variables.
    ks : 1D p-dimensional array, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis.
    r : int, optional
        If r > 0, return a generator yielding estimates for the p-choose-r
        possible combinations of length r from the op variables.

    Returns
    -------
    float
        Entropy estimate

    """
    from itertools import combinations

    ar = _as_data_array(ar, axis=axis)
    p, n = ar.shape

    if ks is None:
        ks = numpy.array([len(numpy.unique(v)) for v in ar])
    else:
        try:
            if len(ks) == p:
                ks = numpy.array(ks)
            else:
                raise ValueError("k should have len %s" % p)
        except TypeError:
            raise

    entropy_estimator = Entropy()
    if r == 0:
        counts = histogram(ar, axis=1)
        k = numpy.prod(ks)
        return entropy_estimator(counts, k=k)
    else:
        counts_combinations = histogram(ar, axis=1, r=r)
        alphabet_size_combinations = (numpy.prod(x)
                                      for x in combinations(ks, r=r))
        return (entropy_estimator(c, k=k) for c, k in
                zip(counts_combinations, alphabet_size_combinations))
