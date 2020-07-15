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
from ndd.counter import Counts
from ndd.exceptions import AlphaError, CardinalityError, NddError
from ndd.package_setup import subclasses

logger = logging.getLogger(__name__)

__all__ = [
    'EntropyEstimator',
    'Plugin',
    'MillerMadow',
    'WolpertWolf',
    'NSB',
    'Grassberger',
    'AsymptoticNSB',
    'AutoEstimator',
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

    @wraps(fit_function)
    def wrapper(obj, nk, k=None, zk=None):
        nk = numpy.asarray(nk)
        if zk is not None:
            zk = numpy.asarray(zk)
        k = obj.check_k(k)
        return fit_function(obj, nk, k=k, zk=zk)

    return wrapper


def guess_alphabet_size(nk, zk=None, eps=1.e-3):
    """Guess a reasonable value for the cardinality."""
    nsb = NSB()
    asym = AsymptoticNSB()
    multiplier = 10
    dk = numpy.log(multiplier)
    if zk is not None:
        k1 = numpy.sum(zk)
    else:
        k1 = numpy.sum([1 for n in nk if n > 0])
        # k1 = k1 // 2
    if not k1:
        k1 = 1
    h0 = nsb(nk=nk, k=k1, zk=zk)
    try:
        hasym = asym(nk=nk, zk=zk)
    except NddError:
        hasym = None  # no coincidences
    for _ in range(40):
        k1 = round(k1 * multiplier)
        h1 = nsb(nk, k=k1, zk=zk)
        dh = (h1 - h0) / dk
        if dh < eps:
            break
        if hasym and h1 >= hasym:  # should return hasym
            raise NddError
        h0 = h1
    return round(k1 / numpy.sqrt(multiplier))  # midpoint value


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
        self.err_ = numpy.float('inf')
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
            if k is None:
                k = numpy.sum(zk[nk > 0])
            if k == 1:
                self.estimate_, self.err_ = PZERO, PZERO
                return self
            if self.alpha:
                self.estimate_ = ndd.fnsb.pseudo_from_multiplicities(
                    nk, zk, k, self.alpha)
            else:
                self.estimate_ = ndd.fnsb.plugin_from_multiplicities(nk, zk)
        else:
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


class PMFPlugin(EntropyEstimator):
    """Entropy from probability mass function array."""

    @check_input
    def fit(self, nk, k=None, zk=None):
        """
        Parameters
        ----------
        nk : array-like
            Probabilities over a set of bins.

        Returns
        -------
        float
            Entropy estimate.

        """
        self.estimate_ = ndd.fnsb.pmf_plugin(nk)
        self.err_ = 0
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

        Notes
        -----
        @article{miller1955note,
          title={Note on the bias of information estimates},
          author={Miller, George},
          journal={Information theory in psychology: Problems and methods},
          year={1955},
          publisher={Free Press}
        }

        """
        plugin = Plugin()
        if zk is not None:
            k = numpy.sum(zk[nk > 0])
            n = numpy.sum(nk * zk)
            self.estimate_ = plugin(nk, k=k, zk=zk) + 0.5 * (k - 1) / n
        else:
            k = numpy.sum(nk > 0)
            n = numpy.sum(nk)
            self.estimate_ = plugin(nk) + 0.5 * (k - 1) / n
        return self


class WolpertWolf(EntropyEstimator):
    """
    Wolpert-Wolf entropy estimator.

    Single Dirichlet prior with concentration parameter `alpha`.

    Parameters
    ----------
    alpha : float
        Concentration parameter. alpha > 0.0.
        If alpha is passed, use a single Dirichlet prior

    Notes
    -----
    @article{wolpert1995estimating,
      title={Estimating functions of probability distributions from a finite set of samples},
      author={Wolpert, David H and Wolf, David R},
      journal={Physical Review E},
      volume={52},
      number={6},
      pages={6841},
      year={1995},
      publisher={APS}
    }

    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = self.check_alpha(alpha)

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
            raise NddError('Wolper-Wolf estimator needs k')
        if k == 1:
            self.estimate_, self.err_ = PZERO, PZERO
            return self

        if zk is not None:
            self.estimate_, self.err_ = ndd.fnsb.ww_from_multiplicities(
                nk, zk, k, self.alpha)
        else:
            self.estimate_, self.err_ = ndd.fnsb.ww(nk, k, self.alpha)
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

    Notes
    -----
    @inproceedings{nemenman2002entropy,
      title={Entropy and inference, revisited},
      author={Nemenman, Ilya and Shafee, Fariel and Bialek, William},
      booktitle={Advances in neural information processing systems},
      pages={471--478},
      year={2002}
    }

    @article{nemenman2004entropy,
      title={Entropy and information in neural spike trains: Progress on the sampling problem},
      author={Nemenman, Ilya and Bialek, William and Van Steveninck, Rob De Ruyter},
      journal={Physical Review E},
      volume={69},
      number={5},
      pages={056111},
      year={2004},
      publisher={APS}
    }

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
            estimator = WolpertWolf(self.alpha).fit(nk=nk, k=k, zk=zk)
            self.estimate_ = estimator.estimate_
            self.err_ = estimator.err_
        return self


class AsymptoticNSB(EntropyEstimator):
    """
    Asymptotic NSB estimator for countably infinite distributions (or with
    unknown cardinality).

    Specifical for the strongly under-sampled regime (k/N approx. 1, where k
    is the number of distinct symbols in the samples and N the number of
    samples)

    Notes
    -----
    @article{nemenman2004entropy,
      title={Entropy and information in neural spike trains: Progress on the sampling problem},
      author={Nemenman, Ilya and Bialek, William and Van Steveninck, Rob De Ruyter},
      journal={Physical Review E},
      volume={69},
      number={5},
      pages={056111},
      year={2004},
      publisher={APS}
    }

    @article{nemenman2011coincidences,
      title={Coincidences and estimation of entropies of random variables with large cardinalities},
      author={Nemenman, Ilya},
      journal={Entropy},
      volume={13},
      number={12},
      pages={2013--2023},
      year={2011},
      publisher={Molecular Diversity Preservation International}
    }

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
        counts = Counts(nk=nk, zk=zk)

        if not counts.coincidences:
            raise NddError('AsymptoticNSB estimator: no coincidences '
                           'in the data.')
        if counts.sampling_ratio > 0.1:
            logger.info('The AsymptoticNSB estimator should only be used '
                        'in the under-sampled regime.')
        if k == 1:
            self.estimate_, self.err_ = PZERO, PZERO
            return self

        self.estimate_ = (euler_gamma - numpy.log(2) +
                          2.0 * numpy.log(counts.n) -
                          ndd.fnsb.gamma0(counts.coincidences))
        self.err_ = numpy.sqrt(ndd.fnsb.gamma1(counts.coincidences))
        return self


class Grassberger(EntropyEstimator):
    """Grassberger's aymptotic bias coorection estimator.

    see equation 35 in:
    https://arxiv.org/pdf/physics/0307138.pdf

    Notes
    -----
    @article{grassberger2003entropy,
      title={Entropy estimates from insufficient samplings},
      author={Grassberger, Peter},
      journal={arXiv preprint physics/0307138},
      year={2003}
    }

    """

    @staticmethod
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
        gg = self.g_series()  # init the G series
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


class AutoEstimator(EntropyEstimator):
    """Select the best estimator for the input data."""

    def __init__(self):
        super().__init__()
        self.estimator = None
        self.k = None

    def guess(self, nk, k=None, zk=None):
        """Select the best estimator given arguments.

        Returns
        -------
        k, estimator

        """

        if k is not None:  # has k?
            self.k = k
            self.estimator = NSB()
            return

        counts = Counts(nk, zk=zk)

        if not counts.coincidences:  # has coincidences?
            logging.warning(
                'Insufficient data (no coincidences found in counts). '
                'Return plugin estimate.')
            self.k = None
            self.estimator = Plugin()  # else Plugin estimator
            return

        if counts.sampling_ratio < 0.1:  # is strongly under-sampled?
            self.k = None
            self.estimator = AsymptoticNSB()
            return

        self.k = guess_alphabet_size(nk=nk,
                                     zk=zk)  # guess a reasonable value for k
        self.estimator = NSB()

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

        """

        self.guess(nk=nk, k=k, zk=zk)
        self.estimator.fit(nk=nk, k=self.k, zk=zk)
        self.estimate_ = self.estimator.estimate_
        self.err_ = self.estimator.err_
        return self


estimators = subclasses(EntropyEstimator)
