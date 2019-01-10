# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""Base classes module."""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)
import numpy
from ndd.base import BaseEstimator
import ndd._nsb


# TODO: docstrings
class Entropy(BaseEstimator):
    def __init__(self, alpha=None, plugin=False):

        self.estimate = None
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

    def __call__(self, *args, **kwargs):
        """Return estimate from input data. Delegate to fit."""
        self.fit(*args, **kwargs)
        return self.estimate


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


def is_pmf(a):
    a = numpy.float64(a)
    not_negative = numpy.all(a >= 0)
    normalized = numpy.isclose(sum(a), 1.0)
    return not_negative and normalized
