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
from ndd.base import BaseEstimator, EntropyEstimatorMixin


# TODO: docstrings
class Entropy(EntropyEstimatorMixin, BaseEstimator):
    def __init__(self, alpha=None, plugin=False):

        self.estimate = None
        self.std = None
        self._estimator = None

        self.alpha = alpha
        self.plugin = plugin

    def fit(self, pk, k=None):
        """
        pk : array_like
            The number of occurrences of a set of bins.

        k : int, optional
            Number of bins. k >= len(pk).
            Float values are valid input for whole numbers (e.g. k=1.e3).
            Defaults to len(pk).

        """
        if k == 1:  # single bin
            self.estimate = self.std = 0.0
        else:
            self.estimate, self.std = self.estimator(pk, k)
        return self

    def __call__(self, *args, **kwargs):
        """Return estimate from input data. Delegate to fit."""
        return self.fit(*args, **kwargs).estimate


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
            Number of bins. k >= len(pk).
            Float values are valid input for whole numbers (e.g. k=1.e3).
            Defaults to len(pk).

        """
        if len(self.log_qk) != len(pk):
            raise ValueError('qk and pk must have the same length.')

        if k == 1:  # single bin
            self.estimate = self.std = 0.0
        else:
            self.estimate, self.std = self.estimator(pk, k)
        self.estimate -= numpy.sum(pk * self.log_qk)
        return self


class JSDivergence(Entropy):
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
        pk = numpy.int32(pk)
        if pk.ndim != 2:
            raise ValueError('counts must be 2D.')
        ws = numpy.float64(pk.sum(axis=1))
        ws /= ws.sum()
        if k == 1:  # single bin
            self.estimate = 0.0
        else:
            self.estimate = self.estimator(pk.sum(axis=0), k)[0] - sum(
                ws[i]*self.estimator(x, k)[0] for i, x in enumerate(pk))
        return self


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
