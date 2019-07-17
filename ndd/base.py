# -*- coding: utf-8 -*-
# Author: Simone Marsili
# License: BSD 3 clause
# pylint: disable=c-extension-no-member
"""Base EntropyEstimator class."""
import abc
import logging

import numpy

import ndd.fnsb
from ndd.base_estimator import BaseEstimator
from ndd.exceptions import AlphaError, CardinalityError, CountsError

logger = logging.getLogger(__name__)


class EntropyEstimator(BaseEstimator, abc.ABC):
    """
    Base class for entropy estimators.

    Attributes
    ----------
    estimate_ : float
        Entropy estimate
    err_ : float or None
        A measure of uncertainty in the estimate. None if not available.

    """

    def __init__(self):
        self.estimate_ = None
        self.err_ = None

    def __call__(self, *args, **kwargs):
        """Fit and return the estimated value."""
        return self.fit(*args, **kwargs).estimate_

    @staticmethod
    def check_alpha(a):
        """Check concentration parameter/#pseudocount.

        Parameters
        ----------
        a : positive number
            Concentration parameter or num. pseudocounts

        Returns
        -------
        a : float64

        Raises
        ------
        AlphaError
            If a is not numeric or negative.

        """
        if a is None:
            raise AlphaError('%s: not a valid alpha value.')
        try:
            a = numpy.float64(a)
        except ValueError:
            raise AlphaError('alpha (%r) should be numeric.' % a)
        if a < 0:
            raise AlphaError('Negative alpha value: %r' % a)
        return a

    @staticmethod
    def check_pk(a):
        """
        Raises
        ------
        CountsError
            If pk is not a valid array of counts.

        """

        a = numpy.float64(a).flatten()
        not_integers = not numpy.all([x.is_integer() for x in a])
        negative = numpy.any([a < 0])
        if not_integers:
            raise CountsError('counts array has non-integer values')
        if negative:
            raise CountsError('counts array has negative values')
        return numpy.int32(a)

    @staticmethod
    def check_k(k):
        """
        if k is an integer, just check
        if an array set k = prod(k)
        if None, return

        Raises
        ------
        CardinalityError
            If k is not valid (wrong type, negative, too large...)

        """
        MAX_LOGK = 150 * numpy.log(2)

        if k is None:
            return k

        try:
            k = numpy.float64(k)
        except ValueError:
            raise CardinalityError('%s: not a valid cardinality')
        if k.ndim:
            # if k is a sequence, set k = prod(k)
            if k.ndim > 1:
                raise CardinalityError('k must be a scalar or 1D array')
            logk = numpy.sum(numpy.log(x) for x in k)
            if logk > MAX_LOGK:
                # too large a number; backoff to n_bins?
                # TODO: log warning
                raise CardinalityError('k (%r) larger than %r' %
                                       (numpy.exp(logk), numpy.exp(MAX_LOGK)))
            k = numpy.prod(k)
        else:
            # if a scalar check size
            if numpy.log(k) > MAX_LOGK:
                raise CardinalityError('k (%r) larger than %r' %
                                       (k, numpy.exp(MAX_LOGK)))
        if not k.is_integer():
            raise CardinalityError('k (%s) should be a whole number.' % k)

        return k

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
        pk = self.check_pk(pk)
        k = self.check_k(k)

        self.estimate_, self.err_ = self.estimator(pk, k)
        return self

    @abc.abstractmethod
    def estimator(self, pk, k):
        """Entropy estimator function.

        Return an entropy estimate given counts and the sample space size.

        Parameters
        ----------
        pk : array-like
            An array of non-negative integers (counts array).
        k  : int or sequence or None
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


class EntropyEstimatorMixin:
    """Mixin class for EntropyEstimator.

    Methods for estimator selection and estimation.

    """

    @staticmethod
    def _plugin_estimator(pk, k):
        return ndd.fnsb.plugin(pk, k), None

    @staticmethod
    def _pseudocounts_estimator(pk, k, alpha):
        return ndd.fnsb.pseudo(pk, k, alpha), None

    @staticmethod
    def _ww_estimator(pk, k, alpha):
        return ndd.fnsb.dirichlet(pk, k, alpha), None

    @staticmethod
    def _nsb_estimator(pk, k):
        return ndd.fnsb.nsb(pk, k)

    @property
    def estimator(self):
        """
        Entropy estimator function.

        Return the object estimator.
        The four possible entropy estimator functions are:
        plugin, plugin with pseudocounts, Wolpert-Wolf (WW) and
        Nemenman-Shafee-Bialek (NSB).

        """

        if self._estimator is None:
            if self.plugin:
                if self.alpha is None:
                    self._estimator = self._plugin_estimator
                else:

                    def pseudocounts_estimator(pk, k):
                        return self._pseudocounts_estimator(pk, k, self.alpha)

                    self._estimator = pseudocounts_estimator
            else:
                if self.alpha is None:
                    self._estimator = self._nsb_estimator
                else:

                    def ww_estimator(pk, k):
                        return self._ww_estimator(pk, k, self.alpha)

                    self._estimator = ww_estimator
        return self._estimator

    def entropy_estimate(self, pk, k):
        """
        Return an entropy estimate given counts and the sample space size.

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
        pk = self._check_pk(pk)
        if k is None:
            k = len(pk)
        k = self._check_k(k)

        zero = numpy.float64(0)
        if k == 1:
            return zero, zero
        return self.estimator(pk, k)

    @staticmethod
    def _check_pk(a):
        """
        Raises
        ------
        CountsError
            If pk is not a valid array of counts.

        """

        a = numpy.float64(a).flatten()
        not_integers = not numpy.all([x.is_integer() for x in a])
        negative = numpy.any([a < 0])
        if not_integers:
            raise CountsError('counts array has non-integer values')
        if negative:
            raise CountsError('counts array has negative values')
        return numpy.int32(a)

    @staticmethod
    def _check_k(k):
        """
        if k is None, set k = number of bins
        if k is an integer, just check
        ik an array set k = prod(k)

        Raises
        ------
        CardinalityError
            If k is not valid (wrong type, negative, too large...)

        """
        MAX_LOGK = 150 * numpy.log(2)

        try:
            k = numpy.float64(k)
        except ValueError:
            raise CardinalityError('%s: not a valid cardinality')
        if k.ndim:
            # if k is a sequence, set k = prod(k)
            if k.ndim > 1:
                raise CardinalityError('k must be a scalar or 1D array')
            logk = numpy.sum(numpy.log(x) for x in k)
            if logk > MAX_LOGK:
                # too large a number; backoff to n_bins?
                # TODO: log warning
                raise CardinalityError('k (%r) larger than %r' %
                                       (numpy.exp(logk), numpy.exp(MAX_LOGK)))
            k = numpy.prod(k)
        else:
            # if a scalar check size
            if numpy.log(k) > MAX_LOGK:
                raise CardinalityError('k (%r) larger than %r' %
                                       (k, numpy.exp(MAX_LOGK)))
        if not k.is_integer():
            raise CardinalityError('k (%s) should be a whole number.' % k)
        return k


class EntropyBasedEstimator(BaseEstimator, abc.ABC):
    """Extend the BaseEstimator to estimators of entropy-derived quantities.

    Specific estimators should extend the EntropyBasedEstimator class with
    a fit() method. The fit() method must set the estimator object attributes
    estimate_ and err_ (using the entropy_estimate method).

    Parameters
    ----------
    estimator : EntropyEstimator object

    Attributes
    ----------
    estimate_ : float
        Entropy estimate
    err_ : float or None
        A measure of uncertainty in the estimate. None if not available.

    """

    def __init__(self, estimator):
        self._estimator = estimator
        self._algorithm = None

        self.estimate_ = None
        self.err_ = None

    def __call__(self, *args, **kwargs):
        """Fit and return the estimated value."""
        return self.fit(*args, **kwargs).estimate_

    @property
    def estimator(self):
        """EntropyEstimator object."""
        return self._estimator

    @estimator.setter
    def estimator(self, obj):
        """Estimator setter."""
        if isinstance(obj, EntropyEstimator):
            self._estimator = obj
        else:
            raise TypeError('Not a EntropyEstimator object.')

    @property
    def algorithm(self):
        """Estimator function name."""
        return self.estimator.__class__.__name__

    @abc.abstractmethod
    def fit(self, pk, k=None):
        """Set the estimated parameters."""
