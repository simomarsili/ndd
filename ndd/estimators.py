# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Base classes module."""
import logging

import numpy
from numpy import PZERO, euler_gamma  # pylint: disable=no-name-in-module

import ndd.fnsb
from ndd.base import EntropyBasedEstimator, EntropyEstimator
from ndd.exceptions import CountsError

logger = logging.getLogger(__name__)

__all__ = ['Entropy', 'JSDivergence']


class Plugin(EntropyEstimator):
    """Plugin entropy estimator."""

    def estimator(self, pk, k=None):
        """Estimator definition."""
        k = len(pk)
        if k == 1:
            return PZERO, PZERO
        return ndd.fnsb.plugin(pk, k), None


class PseudoPlugin(EntropyEstimator):
    """Plugin estimator with pseudoconts."""

    def __init__(self, alpha):
        super().__init__()
        alpha = self.check_alpha(alpha)
        self.alpha = alpha

    def estimator(self, pk, k=None):
        """Estimator definition."""
        if k is None:
            k = len(pk)
        if k == 1:
            return PZERO, PZERO
        return ndd.fnsb.pseudo(pk, k, self.alpha), None


class Miller(EntropyEstimator):
    """Miller entropy estimator class."""

    def estimator(self, pk, k=None):
        """Estimator definition.

        If k is None, set k = #bins with frequency > 0
        (Miller-Madow).
        """
        if k is None:
            k = sum(pk > 0)

        plugin = Plugin()
        n = sum(pk)
        return plugin(pk) + 0.5 * (k - 1) / n, None


class WolpertWolf(EntropyEstimator):
    """Pseudoconts entropy estimator class."""

    def __init__(self, alpha):
        super().__init__()
        alpha = self.check_alpha(alpha)
        self.alpha = alpha

    def estimator(self, pk, k):
        """Estimator definition."""
        if k is None:
            raise ValueError('Wolpert-Wolf estimator needs k')
        if k == 1:
            return PZERO, PZERO
        return ndd.fnsb.dirichlet(pk, k, self.alpha), None


class NSB(EntropyEstimator):
    """NSB entropy estimator class."""

    def estimator(self, pk, k):
        """Estimator definition."""
        if k is None:
            raise ValueError('NSB estimator needs k')
        if k == 1:
            return PZERO, PZERO
        return ndd.fnsb.nsb(pk, k)


class NSBAsymptotic(EntropyEstimator):
    """NSB entropy estimator class."""

    def estimator(self, pk, k=None):
        """Estimator definition."""
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
            return PZERO, PZERO
        return (euler_gamma - numpy.log(2) + 2.0 * numpy.log(n) -
                digamma(delta)), None


class Grassberger(EntropyEstimator):
    """Grassberger 1988 estimator class.

    see:
    http://hornacek.coa.edu/dave/Junk/entropy.estimation.pdf
    https://www.sciencedirect.com/science/article/abs/pii/0375960188901934
    """

    def estimator(self, pk, k=None):
        """Estimator definition."""
        from scipy.special import digamma

        n = sum(pk)

        estimate = n * numpy.log(n)
        for x in pk:
            if not x:
                continue
            estimate -= x * digamma(x) + (1 - 2 * (x % 2)) / (x + 1)
        estimate /= n

        return estimate, None


class Entropy(EntropyBasedEstimator):
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
        self.estimate_, self.err_ = self.entropy_estimate(pk, k)
        return self


class JSDivergence(EntropyBasedEstimator):
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

    """

    def fit(self, pk, k=None):
        """
        Attributes
        ----------
        pk : array_like
            n-by-p array. Different rows correspond to counts from different
            distributions with the same discrete sample space.

        k : int, optional
            Number of bins. k >= p if pk is n-by-p.
            Float values are valid input for whole numbers (e.g. k=1.e3).
            Defaults to pk.shape[1].

        Returns
        -------
        self : object
            Returns the instance itself.

        Raises
        ------
        CountsError
            If pk is not a 2D array.

        """
        pk = numpy.int32(pk)
        if pk.ndim != 2:
            raise CountsError('counts array must be 2D.')
        ws = numpy.float64(pk.sum(axis=1))
        ws /= ws.sum()
        if k == 1:  # single bin
            self.estimate_ = 0.0
        else:
            self.estimate_ = self.entropy_estimate(pk.sum(axis=0), k)[0] - sum(
                ws[i] * self.entropy_estimate(x, k)[0]
                for i, x in enumerate(pk))
        return self
