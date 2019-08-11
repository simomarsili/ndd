# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Classes for entropy estimators."""
import logging
from abc import ABCMeta, abstractmethod

import numpy
from numpy import PZERO, euler_gamma  # pylint: disable=no-name-in-module

import ndd
import ndd.fnsb
from ndd.base import BaseEstimator
from ndd.exceptions import AlphaError, CardinalityError, CountsError, NddError

logger = logging.getLogger(__name__)

__all__ = [
    'EntropyEstimator',
    'Plugin',
    'MillerMadow',
    'Grassberger',
    'NSB',
    'AsymptoticNSB',
]


def check_estimator(estimator):
    """Check that estimator is a valid entropy estimator."""
    if isinstance(estimator, str):
        try:
            estimator_name = estimator
            estimator = getattr(ndd.estimators, estimator_name)()
        except AttributeError:
            raise NddError('%s is not a valid entropy estimator' %
                           estimator_name)
    else:
        estimator_name = type(estimator).__name__

    if estimator_name not in ndd.entropy_estimators:
        raise NddError('%s is not a valid entropy estimator' % estimator_name)

    return estimator, estimator_name


def check_input(fit_function):  # pylint: disable=no-self-argument
    """Check fit input args."""

    def wrapper(obj, pk, k=None):
        pk = obj.check_pk(pk)
        k = obj.check_k(k)
        return fit_function(obj, pk, k=k)

    return wrapper


def g_series():
    """Higher-order function storing terms of the series."""
    GG = {}

    def gterm(n):
        """Sequence of reals for the Grassberger estimator."""
        if n in GG:
            return GG[n]
        if n <= 2:
            if n < 1:
                value = 0.0
            elif n == 1:
                value = -euler_gamma - numpy.log(2.0)
            elif n == 2:
                value = 2.0 + gterm(1)
        else:
            if n % 2 == 0:
                value = gterm(2) + ndd.fnsb.gamma0(
                    (n + 1) / 2) - ndd.fnsb.gamma0(3 / 2)
                # value = ndd.fnsb.gamma0((n + 1) / 2) + numpy.log(2) - 2
            else:
                value = gterm(n - 1)
        GG[n] = value
        return value

    return gterm


# compatible with both Python 2 and 3
# https://stackoverflow.com/a/38668373
ABC = ABCMeta('ABC', (object, ), {'__slots__': ()})


class EntropyEstimator(BaseEstimator, ABC):
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
        self.input_data_ndim = 1

    def __call__(self, pk, k=None):
        """Fit and return the estimated value."""
        return self.fit(pk, k=k).estimate_

    @property
    def algorithm(self):
        """Estimator function name."""
        return self.__class__.__name__

    @staticmethod
    def check_alpha(a):
        """Check concentration parameter/#pseudocount.
        TODO: return None if alpha is None or alpha is 0

        Parameters
        ----------
        a : positive number
            Concentration parameter or num. pseudocounts.

        Returns
        -------
        a : float64

        Raises
        ------
        AlphaError
            If a is not numeric or <=0.

        """
        error_msg = 'alpha must be a positive number (got %r).' % a
        if a is None:
            raise AlphaError(error_msg)
        try:
            a = numpy.float64(a)
        except ValueError:
            raise AlphaError(error_msg)
        if a <= 0:
            raise AlphaError(error_msg)
        return a

    def check_pk(self, a):
        """
        Raises
        ------
        CountsError
            If pk is not a valid array of counts.

        """

        a = numpy.float64(a)
        # check ndim
        # pylint: disable=comparison-with-callable
        if a.ndim != self.input_data_ndim:
            raise CountsError('counts array must be %s-dimensional' %
                              self.input_data_ndim)
        not_integers = not numpy.all([x.is_integer() for x in a.flat])
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
        MAX_LOGK = 200 * numpy.log(2)

        if k is None:
            return k
        try:
            k = numpy.float64(k)
        except ValueError:
            raise CardinalityError('%r is not a valid cardinality' % k)
        if k.ndim:
            # if k is a sequence, set k = prod(k)
            if k.ndim > 1:
                raise CardinalityError('k must be a scalar or 1D array')
            logk = numpy.sum(numpy.log(x) for x in k)
            if logk > MAX_LOGK:
                # too large a number; backoff to n_bins?
                # TODO: log warning
                raise CardinalityError('k must be smaller than 2^40 ')
            k = numpy.prod(k)
        else:
            # if a scalar check size
            if k <= 0:
                print('k: ', k)
                raise CardinalityError('k must be > 0')
            if numpy.log(k) > MAX_LOGK:
                raise CardinalityError('k must be smaller than 2^40 ')
        if not k.is_integer():
            raise CardinalityError('k must be a whole number (got %r).' % k)

        return k

    @abstractmethod
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


class Plugin(EntropyEstimator):
    """Plugin entropy estimator.

    Parameters
    ----------
    alpha : float
        Add alpha pseudocounts to each frequency count. alpha >= 0.
        Defaults to zero pseudocounts (plugin estimator).

    Returns
    -------
    float
        Entropy estimate.

    """

    def __init__(self, alpha=None):
        super(Plugin, self).__init__()
        if alpha:
            self.alpha = self.check_alpha(alpha)
        else:
            self.alpha = None

    @check_input
    def fit(self, pk, k=None):
        """
        Parameters
        ----------
        pk : array-like
            The number of occurrences of a set of bins.
        k : int or array-like, optional
            Total number of bins (including unobserved bins); k >= len(pk).
            A float is a valid input for whole numbers (e.g. k=1.e3).
            If an array, set k = numpy.prod(k). Defaults to len(pk).

        Returns
        -------
        float
            Entropy estimate.

        """
        if k is None:
            k = len(pk)
        if k == 1:
            self.estimate_, self.err_ = PZERO, PZERO
            return self
        if self.alpha:
            self.estimate_ = ndd.fnsb.pseudo(pk, k, self.alpha)
        else:
            self.estimate_ = ndd.fnsb.plugin(pk)
        return self


class MillerMadow(EntropyEstimator):
    """Miller-Madow entropy estimator."""

    @check_input
    def fit(self, pk, k=None):
        """
        Parameters
        ----------
        pk : array-like
            The number of occurrences of a set of bins.
        k : int or array-like, optional
            Alphabet size (the number of bins with non-zero probability).
            A float is a valid input for whole numbers (e.g. k=1.e3).
            If an array, set k = numpy.prod(k).
            Defaults to the number of bins with frequency > 0 (Miller-Madow).

        Returns
        -------
        float
            Entropy estimate.

        """
        if k is None:
            k = sum(pk > 0)

        plugin = Plugin()
        n = sum(pk)
        self.estimate_ = plugin(pk) + 0.5 * (k - 1) / n
        return self


class NSB(EntropyEstimator):
    """Nemenman-Shafee-Bialek (NSB) entropy estimator.

    Parameters
    ----------
    alpha : float, optional
        Concentration parameter. alpha > 0.0.
        If alpha is passed, use a single Dirichlet prior
        (Wolpert-Wolf estimator).
        Default: use a mixture-of-Dirichlets prior (NSB estimator).

    """

    def __init__(self, alpha=None):
        super(NSB, self).__init__()
        if alpha:
            self.alpha = self.check_alpha(alpha)
        else:
            self.alpha = None

    @check_input
    def fit(self, pk, k=None):
        """
        Parameters
        ----------
        pk : array-like
            The number of occurrences of a set of bins.
        k : int or array-like
            Total number of bins (including unobserved bins); k >= len(pk).
            A float is a valid input for whole numbers (e.g. k=1.e3).
            If an array, set k = numpy.prod(k). Defaults to len(pk).
            If k is None, set k = #bins with frequency > 0 (Miller-Madow).

        Returns
        -------
        float
            Entropy estimate.

        """
        if k is None:
            raise ValueError('NSB estimator needs k')
        if k == 1:
            self.estimate_, self.err_ = PZERO, PZERO
            return self
        if self.alpha is None:
            self.estimate_, self.err_ = ndd.fnsb.nsb(pk, k)
        else:
            self.estimate_ = ndd.fnsb.dirichlet(pk, k, self.alpha)
        return self


class AsymptoticNSB(EntropyEstimator):
    """
    Asymptotic NSB estimator for countably infinite distributions.

    Specifical for the under-sampled regime (k/N approx. 1, where k is the
    number of distinct symbols in the samples and N the number of samples)

    See:
    Nemenman2011:
    "Coincidences and estimation of entropies of random variables
    with largecardinalities.", equations 29, 30
    """

    @check_input
    def fit(self, pk, k=None):
        """
        Parameters
        ----------
        pk : array-like
            The number of occurrences of a set of bins.

        Returns
        -------
        float
            Entropy estimate.
        """
        kn = sum(pk > 0)  # number of sampled bins
        n = sum(pk)  # number of samples
        # under-sampled regime when ratio < 0.1 (Nemenman2011)
        delta = n - kn + 1
        ratio = delta / n

        if ratio > 0.1:
            logger.warning('The AsymptoticNSB estimator should be only used '
                           'in the under-sampled regime.')
        if k == 1:
            self.estimate_, self.err_ = PZERO, PZERO
            return self

        self.estimate_ = (euler_gamma - numpy.log(2) + 2.0 * numpy.log(n) -
                          ndd.fnsb.gamma0(delta))
        self.err_ = numpy.sqrt(ndd.fnsb.gamma1(delta))
        return self


class _Grassberger1(EntropyEstimator):
    """Grassberger 1988 estimator.

    see equation 7 in:
    http://hornacek.coa.edu/dave/Junk/entropy.estimation.pdf
    https://www.sciencedirect.com/science/article/abs/pii/0375960188901934
    """

    @check_input
    def fit(self, pk, k=None):  # pylint: disable=unused-argument
        """Estimator definition."""

        n = sum(pk)

        estimate = n * numpy.log(n)
        for x in pk:
            if x:
                estimate -= x * ndd.fnsb.gamma0(x) + (1 - 2 *
                                                      (x % 2)) / (x + 1)
        estimate /= n

        self.estimate_ = estimate
        return self


class Grassberger(EntropyEstimator):
    """Grassberger 2008 estimator.

    see equation 35 in:
    https://arxiv.org/pdf/physics/0307138.pdf
    """

    @check_input
    def fit(self, pk, k=None):  # pylint: disable=unused-argument
        """Estimator definition."""

        n = sum(pk)

        gg = g_series()  # init the G series
        estimate = n * numpy.log(n)
        for x in pk:
            if x:
                estimate -= x * gg(x)
        estimate /= n

        self.estimate_ = estimate
        return self
