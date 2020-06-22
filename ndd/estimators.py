# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Classes for entropy estimators."""
import logging
from abc import ABC, abstractmethod  # python >= 3.4
from functools import wraps

import numpy
from numpy import PZERO, euler_gamma  # pylint: disable=no-name-in-module

import ndd.fnsb
from ndd.base import BaseEstimator
from ndd.exceptions import AlphaError, CardinalityError, CountsError, NddError
from ndd.package_setup import subclasses

logger = logging.getLogger(__name__)

__all__ = [
    'EntropyEstimator',
    'Plugin',
    'MillerMadow',
    'Grassberger',
    'NSB',
    'AsymptoticNSB',
]


def sampling_ratio(nk, zk=None):
    """Undersampled regime is defined for sampling ratio < 0.1"""
    nk = numpy.asarray(nk)
    zk = numpy.asarray(zk) if zk is not None else zk
    if zk is not None:
        kn = numpy.sum(zk[nk > 0])
        n = numpy.sum(zk * nk)
    else:
        kn = numpy.sum(nk > 0)  # slow
        n = numpy.sum(nk)
    delta = n - kn
    # ratio = delta / (n + 1)
    ratio = delta / n
    # store info as function attributes
    sampling_ratio.n = n
    sampling_ratio.kn = kn
    sampling_ratio.delta = delta
    sampling_ratio.undersampled = ratio < 0.1
    sampling_ratio.coincidences = delta > 0
    return ratio


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

    @wraps(fit_function)
    def wrapper(obj, nk, k=None, zk=None):
        nk = obj.check_nk(nk)
        if zk is not None:
            zk = obj.check_nk(zk)
        k = obj.check_k(k)
        return fit_function(obj, nk, k=k, zk=zk)

    return wrapper


def g_series():
    """Higher-order function storing terms of the series."""
    GG = {}
    gamma0 = ndd.fnsb.gamma0
    log_two = numpy.log(2.0)

    def gterm(n):
        """Sequence of reals for the Grassberger estimator."""
        if n in GG:
            return GG[n]
        if n <= 2:
            if n < 1:
                value = 0.0
            elif n == 1:
                value = -euler_gamma - log_two
            elif n == 2:
                value = 2.0 + gterm(1)
        else:
            if n % 2 == 0:
                value = gamma0((n + 1) / 2) + log_two
            else:
                value = gterm(n - 1)
        GG[n] = value
        return value

    return gterm


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

    def __call__(self, nk, k=None, zk=None):
        """Fit and return the estimated value."""
        return self.fit(nk, k=k, zk=zk).estimate_

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

    @staticmethod
    def check_nk(a):
        """
        Convert the array of counts to int32.

        Raises
        ------
        CountsError
            If nk is not a valid array of counts.

        """
        a = numpy.int32(a)
        negative = numpy.any([a < 0])
        if negative:
            raise CountsError('counts array has negative values')
        return a

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
                raise CardinalityError('k is too large (%e).'
                                       'Must be < 2^200 ' % numpy.exp(logk))
            k = numpy.prod(k)
        else:
            # if a scalar check size
            if k <= 0:
                raise CardinalityError('k must be > 0 (%r)' % k)
            if numpy.log(k) > MAX_LOGK:
                raise CardinalityError('k is too large (%e).'
                                       'Must be < 2^200 ' % k)
        if not k.is_integer():
            raise CardinalityError('k must be a whole number (got %r).' % k)

        return k

    @abstractmethod
    def fit(self, nk, k=None, zk=None):
        """
        Compute an entropy estimate from nk.

        Parameters
        ----------
        nk : array_like, shape (n_bins,)
            The number of occurrences of a set of bins.
        k : int, optional
            Number of bins. k >= len(nk).
            Float values are valid input for whole numbers (e.g. k=1.e3).
            Defaults to sum(nk > 0).
        zk : array_like, optional
            Counts distribution or "multiplicities". If passed, nk contains
            the observed counts values.

        Returns
        -------
        self : object
            Returns the instance itself.

        """


class Plugin(EntropyEstimator):
    """Plugin (maximum likelihood) entropy estimator.

    Insert the maximum likelihood estimate of the PMF from empirical
    frequencies over bins into the entropy definition.
    For alpha > 0, the estimate depends on k (the alphabet size).

    Parameters
    ----------
    alpha : float
        Add alpha pseudocounts to the each frequency count. alpha >= 0.
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
    def fit(self, nk, k=None, zk=None):
        """
        Parameters
        ----------
        nk : array-like
            The number of occurrences of a set of bins.
        k : int or array-like, optional
            Alphabet size (the number of bins with non-zero probability).
            Must be >= len(nk). A float is a valid input for whole numbers
            (e.g. k=1.e3). If an array, set k = numpy.prod(k).
            Default: k = sum(nk > 0)
        zk : array_like, optional
            Counts distribution or "multiplicities". If passed, nk contains
            the observed counts values.

        Returns
        -------
        float
            Entropy estimate.

        """
        if zk is not None:
            raise NotImplementedError('%s estimator takes counts as input' %
                                      self.__class__.__name__)
        if k is None:
            k = numpy.sum(nk > 0)
        if k == 1:
            self.estimate_, self.err_ = PZERO, PZERO
            return self
        if self.alpha:
            self.estimate_ = ndd.fnsb.pseudo(nk, k, self.alpha)
        else:
            self.estimate_ = ndd.fnsb.plugin(nk)
        return self


class MillerMadow(EntropyEstimator):
    """Miller-Madow entropy estimator."""

    @check_input
    def fit(self, nk, k=None, zk=None):
        """
        Parameters
        ----------
        nk : array-like
            The number of occurrences of a set of bins.

        Returns
        -------
        float
            Entropy estimate.

        """
        if zk is not None:
            raise NotImplementedError('%s estimator takes counts as input' %
                                      self.__class__.__name__)
        k = numpy.sum(nk > 0)

        plugin = Plugin()
        n = numpy.sum(nk)
        self.estimate_ = plugin(nk) + 0.5 * (k - 1) / n
        return self


class NSB(EntropyEstimator):
    """
    Nemenman-Shafee-Bialek (NSB) entropy estimator.

    The estimate depends on k (the alphabet size).

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
    def fit(self, nk, k=None, zk=None):
        """
        Parameters
        ----------
        nk : array-like
            The number of occurrences of a set of bins.
        k : int or array-like
            Alphabet size (the number of bins with non-zero probability).
            Must be >= len(nk). A float is a valid input for whole numbers
            (e.g. k=1.e3). If an array, set k = numpy.prod(k).
        zk : array_like, optional
            Counts distribution or "multiplicities". If passed, nk contains
            the observed counts values.

        Returns
        -------
        self : object

        Raises
        ------
        NddError
            If k is None.

        """
        if k is None:
            raise NddError('NSB estimator needs k')
        if k == 1:
            self.estimate_, self.err_ = PZERO, PZERO
            return self

        if self.alpha is None:
            if zk is not None:
                self.estimate_, self.err_ = ndd.fnsb.nsb_from_multiplicities(
                    nk, zk, k)
            else:
                self.estimate_, self.err_ = ndd.fnsb.nsb(nk, k)
        else:  # wolpert-wolf estimator
            if zk is not None:
                self.estimate_, self.err_ = ndd.fnsb.ww_from_multiplicities(
                    nk, zk, k, self.alpha)
            else:
                self.estimate_, self.err_ = ndd.fnsb.ww(nk, k, self.alpha)
        return self


class AsymptoticNSB(EntropyEstimator):
    """
    Asymptotic NSB estimator for countably infinite distributions (or with
    unknown cardinality).

    Specifical for the strongly under-sampled regime (k/N approx. 1, where k
    is the number of distinct symbols in the samples and N the number of
    samples)

    See:
    Nemenman2011:
    "Coincidences and estimation of entropies of random variables
    with largecardinalities.", equations 29, 30

    """

    @check_input
    def fit(self, nk, k=None, zk=None):
        """
        Parameters
        ----------
        nk : array-like
            The number of occurrences of a set of bins.

        Returns
        -------
        float
            Entropy estimate.

        Raises
        ------
        NddError
            No coincindences.

        """
        if zk is not None:
            raise NotImplementedError('%s estimator takes counts as input' %
                                      self.__class__.__name__)
        ratio = sampling_ratio(nk=nk, zk=zk)
        delta = sampling_ratio.delta
        n = sampling_ratio.n
        coincidences = sampling_ratio.n - sampling_ratio.kn

        if not coincidences:
            raise NddError('AsymptoticNSB estimator: no coincidences '
                           'in the data.')
        if ratio > 0.1:
            logger.info('The AsymptoticNSB estimator should only be used '
                        'in the under-sampled regime.')
        if k == 1:
            self.estimate_, self.err_ = PZERO, PZERO
            return self

        self.estimate_ = (euler_gamma - numpy.log(2) + 2.0 * numpy.log(n) -
                          ndd.fnsb.gamma0(delta))
        self.err_ = numpy.sqrt(ndd.fnsb.gamma1(delta))
        return self


class Grassberger(EntropyEstimator):
    """Grassberger 2003 estimator.

    see equation 35 in:
    https://arxiv.org/pdf/physics/0307138.pdf

    """

    @check_input
    def fit(self, nk, k=None, zk=None):  # pylint: disable=unused-argument
        """
        Parameters
        ----------
        nk : array-like
            The number of occurrences of a set of bins.

        Returns
        -------
        float
            Entropy estimate.

        """
        gg = g_series()  # init the G series
        estimate = 0

        if zk is not None:
            n = numpy.sum(nk * zk)
            for j, x in enumerate(nk):
                if x:
                    estimate -= zk[j] * x * gg(x)
        else:
            n = numpy.sum(nk)
            for x in nk:
                if x:
                    estimate -= x * gg(x)
        estimate = numpy.log(n) - estimate / n

        self.estimate_ = estimate
        return self


estimators = subclasses(EntropyEstimator)
