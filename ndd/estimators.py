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

__all__ = ['Entropy', 'KLDivergence', 'JSDivergence']


# TODO: docstrings
class Entropy(EntropyEstimator):
    """Entropy estimator class.

    Default: use the NSB estimator function.

    Parameters
    ----------
    alpha : float, optional
        If not None: Wolpert-Wolf estimator (fixed alpha).
        A single Dirichlet prior with concentration parameter alpha.
        alpha > 0.0.
    plugin : boolean, optional
        If True: 'plugin' estimator.
        The discrete distribution is estimated from the empirical frequencies
        over bins and inserted into the entropy definition (plugin estimator).
        If alpha is passed in combination with plugin=True, add
        alpha pseudocounts to each frequency count (pseudocount estimator).

    Attributes
    ----------
    estimator : estimator function
        The four possible entropy estimator functions are: plugin, plugin with
        pseudocounts, Wolpert-Wolf (WW) and Nemenman-Shafee-Bialek (NSB).

    """

    def fit(self, pk, k=None):
        """
        Compute an entropy estimate from pk.

        Parameters
        ----------
        pk : array_like, shape (n_bins,)
            The number of occurrences of a set of bins.
        k : int, optional
            Number of bins. k >= len(pk).
            Float values are valid input for whole numbers (e.g. k=1.e3).
            Defaults to len(pk).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if k == 1:  # single bin
            self.estimate_ = self.err_ = 0.0
        else:
            self.estimate_, self.err_ = self.estimator(pk, k)
        return self


class KLDivergence(EntropyEstimator):
    """Kullback-Leibler divergence estimator class.

    Default: use the NSB estimator function.

    Parameters
    ----------
    alpha : float, optional
        If alpha is not None: Wolpert-Wolf estimator (fixed alpha).
        A single Dirichlet prior with concentration parameter alpha.
        alpha > 0.0.
    plugin : boolean, optional
        If True: 'plugin' estimator.
        The discrete distribution is estimated from the empirical frequencies
        over bins and inserted into the entropy definition (plugin estimator).
        If alpha is passed in combination with plugin=True, add
        alpha pseudocounts to each frequency count (pseudocount estimator).

    Attributes
    ----------
    estimator : estimator function
        The four possible entropy estimator functions are: plugin, plugin with
        pseudocounts, Wolpert-Wolf (WW) and Nemenman-Shafee-Bialek (NSB).

    """
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
            self.estimate_ = self.err_ = 0.0
        else:
            self.estimate_, self.err_ = self.estimator(pk, k)
        self.estimate_ += numpy.sum(pk * log_qk) / float(sum(pk))
        self.estimate_ = - self.estimate_
        return self


class JSDivergence(EntropyEstimator):
    """Jensen-Shannon divergence estimator class.

    Default: use the NSB estimator function.

    Parameters
    ----------
    alpha : float, optional
        If alpha is not None: Wolpert-Wolf estimator (fixed alpha).
        A single Dirichlet prior with concentration parameter alpha.
        alpha > 0.0.
    plugin : boolean, optional
        If True: 'plugin' estimator.
        The discrete distribution is estimated from the empirical frequencies
        over bins and inserted into the entropy definition (plugin estimator).
        If alpha is passed in combination with plugin=True, add
        alpha pseudocounts to each frequency count (pseudocount estimator).

    Attributes
    ----------
    estimator : estimator function
        The four possible entropy estimator functions are: plugin, plugin with
        pseudocounts, Wolpert-Wolf (WW) and Nemenman-Shafee-Bialek (NSB).

    """

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
            self.estimate_ = 0.0
        else:
            self.estimate_ = self.estimator(pk.sum(axis=0), k)[0] - sum(
                ws[i] * self.estimator(x, k)[0] for i, x in enumerate(pk))
        return self


def is_pmf(a):
    a = numpy.float64(a)
    not_negative = numpy.all(a >= 0)
    normalized = numpy.isclose(sum(a), 1.0)
    return not_negative and normalized
