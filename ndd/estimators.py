# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Classes for entropy estimators."""
import logging
from abc import ABCMeta, abstractmethod

import numpy
from numpy import PZERO, euler_gamma  # pylint: disable=no-name-in-module

import ndd.fnsb
from ndd.base import BaseEstimator
from ndd.exceptions import AlphaError, CardinalityError, CountsError

logger = logging.getLogger(__name__)

__all__ = [
    'EntropyEstimator', 'Plugin', 'Miller', 'WolpertWolf', 'NSB',
    'AsymptoticNSB'
]


def check_input(fit_function):  # pylint: disable=no-self-argument
    """Check fit input args."""

    def wrapper(obj, pk, k=None):
        pk = obj.check_pk(pk)
        k = obj.check_k(k)
        return fit_function(obj, pk, k=k)

    return wrapper


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
        MAX_LOGK = 150 * numpy.log(2)

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
                raise CardinalityError('k must be smaller than 2^150 ')
            k = numpy.prod(k)
        else:
            # if a scalar check size
            if k <= 0:
                print('k: ', k)
                raise CardinalityError('k must be > 0')
            if numpy.log(k) > MAX_LOGK:
                raise CardinalityError('k must be smaller than 2^150 ')
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
        if not alpha:
            self.alpha = PZERO
        else:
            self.alpha = self.check_alpha(alpha)

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
        if self.alpha:
            self.estimate_ = ndd.fnsb.pseudo(pk, k, self.alpha)
        else:
            self.estimate_ = ndd.fnsb.plugin(pk)
        return self


class Miller(EntropyEstimator):
    """Miller entropy estimator."""

    @check_input
    def fit(self, pk, k=None):
        """
        Parameters
        ----------
        pk : array-like
            The number of occurrences of a set of bins.
        k : int or array-like, optional
            Total number of accessible bins (including unobserved bins)
            A float is a valid input for whole numbers (e.g. k=1.e3).
            If an array, set k = numpy.prod(k). Defaults to len(pk).
            If k is None, set k = #bins with frequency > 0 (Miller-Madow).

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


class WolpertWolf(EntropyEstimator):
    """
    Wolpert-Wolf (single Dirichlet prior) estimator.

    See:
    "Estimating functions of probability distributions from a finite set of
    samples"
    Wolpert, David H and Wolf, David R

    Parameters
    ----------
    alpha : float
        Concentration parameter. alpha > 0.0.
    """

    def __init__(self, alpha):
        super(WolpertWolf, self).__init__()
        self.alpha = self.check_alpha(alpha)

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
            raise ValueError('Wolpert-Wolf estimator needs k')
        if k == 1:
            self.estimate_, self.err_ = PZERO, PZERO
        self.estimate_ = ndd.fnsb.dirichlet(pk, k, self.alpha)
        return self


class NSB(EntropyEstimator):
    """NSB entropy estimator."""

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
        self.estimate_, self.err_ = ndd.fnsb.nsb(pk, k)
        return self


class AsymptoticNSB(EntropyEstimator):
    """
    Asymptotic NSB estimator for countably infinite distributions.

    Specifical for the under-sampled regime (k/N approx. 1, where k is the
    number of distinct symbols in the samples and N the number of samples)

    See:
    "Coincidences and estimation of entropies of random variables with large
    cardinalities."
    I. Nemenman.
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
        from scipy.special import digamma
        n = sum(pk)  # #samples
        k1 = sum(pk > 0)  # #sampled bins
        delta = n - k1
        if delta == 0:
            raise ValueError('NSBAsymptotic: No coincidences in data')
        ratio = k1 / n
        if ratio <= 0.9:
            logger.warning('NSB asymptotic should be used in the '
                           'under-sampled regime only.')
        if k == 1:
            self.estimate_, self.err_ = PZERO, PZERO
        self.estimate_ = (euler_gamma - numpy.log(2) + 2.0 * numpy.log(n) -
                          digamma(delta))
        return self


class Grassberger(EntropyEstimator):
    """Grassberger 1988 estimator.

    see:
    http://hornacek.coa.edu/dave/Junk/entropy.estimation.pdf
    https://www.sciencedirect.com/science/article/abs/pii/0375960188901934
    """

    @check_input
    def fit(self, pk, k=None):  # pylint: disable=unused-argument
        """Estimator definition."""
        from scipy.special import digamma

        n = sum(pk)

        estimate = n * numpy.log(n)
        for x in pk:
            if not x:
                continue
            estimate -= x * digamma(x) + (1 - 2 * (x % 2)) / (x + 1)
        estimate /= n

        self.estimate_ = estimate
        return self


class UnderWell(EntropyEstimator):
    """Combination of two estimators.

    Combination of an estimator for the under-sampled regime (asymptotic NSB)
    and another for the well-sampled regime (GRassberger)
    """

    @check_input
    def fit(self, pk, k=None):  # pylint: disable=unused-argument
        """Estimator definition."""
        k1 = len(pk > 0)
        n = sum(pk)
        ratio = k1 / n

        under_sampled_estimator = AsymptoticNSB()
        well_sampled_estimator = Grassberger()
        self.estimate_ = (ratio**2 * under_sampled_estimator(pk) +
                          (1 - ratio**2) * well_sampled_estimator(pk))
        return self
