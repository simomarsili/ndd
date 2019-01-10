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
from ndd.base import BaseEstimator


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
        return ndd._nsb.plugin(counts, k), None

    @staticmethod
    def _pseudocounts(counts, k, alpha):
        return ndd._nsb.pseudo(counts, k, alpha), None

    @staticmethod
    def _ww(counts, k, alpha):
        return ndd._nsb.dirichlet(counts, k, alpha), None

    @staticmethod
    def _nsb(counts, k):
        return ndd._nsb.nsb(counts, k)

    def _check_input(self, counts, k):
        counts = self._check_counts(a=counts)
        k = self._check_k(k=k, n_bins=len(counts))
        return counts, k

    @staticmethod
    def _check_counts(a):
        a = numpy.float64(a).flatten()
        not_integers = not numpy.all([x.is_integer() for x in a])
        negative = numpy.any([a < 0])
        if not_integers:
            raise ValueError('counts array has non-integer values')
        if negative:
            raise ValueError('counts array has negative values')
        return numpy.int32(a)

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
        counts, k = self._check_input(counts, k)
        if k == 1:  # single bin
            self.estimate = self.std = 0.0
        else:
            self.estimate, self.std = self.estimator(counts, k)


class KLDivergence(Entropy):
    def __init__(self, qk, alpha=None, plugin=False):
        """qk is a parameter of the estimator; it must be a valid pmf."""
        super().__init__(alpha, plugin)
        if is_pmf(qk):
            self.log_qk = numpy.log(qk)
        else:
            raise ValueError('qk must be a valid PMF')

    def fit(self, pk, k=None):
        """
        pk : array_like
            The number of occurrences of a set of bins.

        k : int, optional
            Number of bins. k >= len(counts).
            Float values are valid input for whole numbers (e.g. k=1.e3).
            Defaults to len(counts).

        """
        pk, k = self._check_input(pk, k)
        if not len(self.log_qk) == len(pk):
            raise ValueError('qk and pk must have the same length.')

        if k == 1:  # single bin
            self.estimate = self.std = 0.0
        else:
            self.estimate, self.std = self.estimator(pk, k)
        self.estimate -= numpy.sum(pk * self.log_qk)


class JSDivergence(Entropy):
    @staticmethod
    def _check_counts(a):
        a = numpy.float64(a)
        if a.ndim != 2:
            raise ValueError('counts must be 2D.')
        not_integers = not numpy.all([x.is_integer() for x in a.flatten()])
        negative = numpy.any([a < 0])
        if not_integers:
            raise ValueError('counts array has non-integer values')
        if negative:
            raise ValueError('counts array has negative values')
        return numpy.int32(a)

    def _check_input(self, pk, k):
        pk = self._check_counts(a=pk)
        k = self._check_k(k=k, n_bins=pk.shape[1])
        return pk, k

    def fit(self, pk, k=None):
        """
        pk : array_like
            n-by-p array. Different rows correspond to counts from different
            distributions with the same discrete sample space.

        k : int, optional
            Number of bins. k >= p if pk is n-by-p.
            Float values are valid input for whole numbers (e.g. k=1.e3).
            Defaults to pk.shape[1].

        """
        pk, k = self._check_input(pk, k)
        n, p = pk.shape
        ws = numpy.float64(pk.sum(axis=1))
        ws /= ws.sum()
        if k == 1:  # single bin
            self.estimate = 0.0
        else:
            self.estimate = self.estimator(pk.sum(axis=0), k)[0] - sum(
                ws[i]*self.estimator(x, k)[0] for i, x in enumerate(pk))


def entropy(pk, k=None, alpha=None, plugin=False, return_std=False):
    """
    Return a Bayesian estimate S' of the entropy of an unknown discrete
    distribution from an input array of counts pk.

    Parameters
    ----------

    pk : array-like
        The number of occurrences of a set of bins.
    k : int or array-like, optional
        Number of bins; k >= len(pk).
        A float value is a valid input for whole numbers (e.g. k=1.e3).
        If an array, set k = numpy.prod(k).
        Defaults to len(counts).
    alpha : float, optional
        If alpha is not None, use a single Dirichlet prior with concentration
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
        (approximated standard deviation over the entropy posterior).
        Only provided if `return_std` is True.

    """

    # pk is an array of counts
    estimator = Entropy(alpha, plugin)
    estimator.fit(pk, k)
    S, err = estimator.estimate, estimator.std

    if numpy.isnan(S) or (err is not None and numpy.isnan(err)):
        raise FloatingPointError("NaN value")

    if return_std:
        return S, err
    else:
        return S


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


def is_pmf(a):
    a = numpy.float64(a)
    not_negative = numpy.all(a >= 0)
    normalized = numpy.isclose(sum(a), 1.0)
    return not_negative and normalized


def _test_JS(n):
    alpha = 0.1
    p = 100
    p1 = numpy.random.dirichlet([alpha]*p)
    p2 = numpy.random.dirichlet([alpha]*p)
    pm = 0.5 * (p1 + p2)

    def ee(x):
        y = - x * numpy.log(x)
        return numpy.sum(y[x > 0])
    js = ee(pm) - 0.5 * (ee(p1) + ee(p2))
    c1 = numpy.random.multinomial(n, p1)
    c2 = numpy.random.multinomial(n, p2)
    pk = numpy.stack([c1, c2])
    est = JSDivergence()
    est.fit(pk)
    js1 = est.estimate
    return js, js1
