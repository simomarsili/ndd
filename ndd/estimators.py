# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""Base classes module."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next,
    oct, open, pow, round, super, filter, map, zip)
import numpy
from ndd.base import EntropyEstimator


# TODO: docstrings
class Entropy(EntropyEstimator):
    """Entropy estimator class."""

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


class KLDivergence(EntropyEstimator):
    """Kullback-Leibler divergence estimator class."""

    def fit(self, pk, qk, k=None):
        """
        pk : array_like
            The number of occurrences of a set of bins.

        qk : array_like
            Reference PMF in sum(pk log(pk/qk).
            Must be a valid PMF (non-negative, normalized).

        k : int, optional
            Number of bins. k >= len(pk).
            Float values are valid input for whole numbers (e.g. k=1.e3).
            Defaults to len(pk).

        """
        if is_pmf(qk):
            log_qk = numpy.log(qk)
        else:
            raise ValueError('qk must be a valid PMF')

        if len(log_qk) != len(pk):
            raise ValueError('qk and pk must have the same length.')

        if k == 1:  # single bin
            self.estimate = self.std = 0.0
        else:
            self.estimate, self.std = self.estimator(pk, k)
        self.estimate += numpy.sum(pk * log_qk) / float(sum(pk))
        self.estimate = - self.estimate
        return self


class JSDivergence(EntropyEstimator):
    """Jensen-Shannon divergence estimator class."""

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
                ws[i] * self.estimator(x, k)[0] for i, x in enumerate(pk))
        return self


def _test_JS(n):
    alpha = 0.1
    p = 100
    p1 = numpy.random.dirichlet([alpha] * p)
    p2 = numpy.random.dirichlet([alpha] * p)
    pm = 0.5 * (p1 + p2)

    def ee(x):
        y = -x * numpy.log(x)
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
