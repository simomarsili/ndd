# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""Base EntropyEstimator class."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next,
    oct, open, pow, round, super, filter, map, zip)
import numpy
from ndd.base_estimator import BaseEstimator
import ndd.fnsb


class EntropyEstimatorMixin(object):
    """Mixin class for EntropyEstimator.

    Contains methods to select an estimator and compute an estimates from data.
    """

    def plugin_estimator(self, pk, k):
        return ndd.fnsb.plugin(pk, k), None

    def pseudocounts_estimator(self, pk, k, alpha):
        return ndd.fnsb.pseudo(pk, k, alpha), None

    def ww_estimator(self, pk, k, alpha):
        return ndd.fnsb.dirichlet(pk, k, alpha), None

    def nsb_estimator(self, pk, k):
        return ndd.fnsb.nsb(pk, k)

    def select_estimator(self):
        """
        Return an estimator function for the object.
        Possible estimators are:
        - NSB (Nemenman-Shafee-Bialek)
        - WW (Wolper-Wolf)
        - "plugin"
        - pseudocounts-regularized plugin
        """

        if self.plugin:
            if self.alpha is None:
                return self.plugin_estimator
            else:
                def pseudocounts_estimator(pk, k):
                    return self.pseudocounts_estimator(pk, k, self.alpha)
                return pseudocounts_estimator
        else:
            if self.alpha is None:
                return self.nsb_estimator
            else:
                def ww_estimator(pk, k):
                    return self.ww_estimator(pk, k, self.alpha)
                return ww_estimator

    def estimator(self, pk, k):
        """
        Return an entropy estimate from counts and the size of sample space.

        Parameters
        ----------
        pk : array-like
            An array of non-negative integers (counts array).
        k  : int or sequence
            Size of the sample space.
            Float values are valid input for whole numbers (e.g. k=1.e3).
            If a sequence, set k = numpy.prod(k).

        Returns
        -------
        estimate : float
            Entropy estimate
        err : float or None
            A measure of uncertainty in the estimate. None if not available.

        """
        pk = self.check_pk(pk)
        if k is None:
            k = len(pk)
        k = self.check_k(k)

        return self.estimator_function(pk, k)

    @staticmethod
    def check_pk(a):
        a = numpy.float64(a).flatten()
        not_integers = not numpy.all([x.is_integer() for x in a])
        negative = numpy.any([a < 0])
        if not_integers:
            raise ValueError('counts array has non-integer values')
        if negative:
            raise ValueError('counts array has negative values')
        return numpy.int32(a)

    @staticmethod
    def check_k(k):
        """
        if k is None, set k = number of bins
        if k is an integer, just check
        ik an array set k = prod(k)
        """
        MAX_LOGK = 150 * numpy.log(2)

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
                raise ValueError(
                    'k (%r) larger than %r' % (k, numpy.exp(MAX_LOGK)))
        if not k.is_integer():
            raise ValueError("k (%s) should be a whole number." % k)
        return k


class EntropyEstimator(BaseEstimator, EntropyEstimatorMixin):
    """Extend the BaseEstimator to estimators of entropy-derived quantities.

    Specific estimators should extend the EntropyEstimator class with a fit()
    method. The fit() method must set the estimator object attributes
    estimate and err (using the estimator_function method).

    Parameters
    ----------
    alpha : float, optional
        If not None: Wolpert-Wolf estimator function (fixed alpha).
        A single Dirichlet prior with concentration parameter alpha.
        alpha > 0.0.
    plugin : boolean, optional
        If True: 'plugin' estimator function.
        The discrete distribution is estimated from the empirical frequencies
        over bins and inserted into the entropy definition (plugin estimator).
        If alpha is passed in combination with plugin=True, add
        alpha pseudocounts to each frequency count (pseudocount estimator).

    Attributes
    ----------
    estimate_ : float
        Entropy estimate
    err_ : float or None
        A measure of uncertainty in the estimate. None if not available.

    """

    def __init__(self, alpha=None, plugin=False):
        self.alpha = self.check_alpha(alpha)
        self.plugin = plugin
        self._estimator_function = None
        self._algorithm = None

        self.estimate_ = None
        self.err_ = None

    def __call__(self, *args, **kwargs):
        """Fit and return the estimated value."""
        return self.fit(*args, **kwargs).estimate_

    def check_alpha(self, a):
        if a is None:
            return a
        try:
            a = numpy.float64(a)
        except ValueError:
            raise ValueError('alpha (%r) should be numeric.' % a)
        if a < 0:
            raise ValueError('Negative alpha value: %r' % a)
        return a

    @property
    def estimator_function(self):
        """Entropy estimator function."""
        if self._estimator_function is None:
            self._estimator_function = self.select_estimator()
        return self._estimator_function

    @property
    def algorithm(self):
        """Estimator function name."""
        return self.estimator_function.__name__.split('_')[0]

    def fit(self):
        """Set the estimated parameters."""
        raise NotImplemented
