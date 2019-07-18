# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Base classes module."""
import logging

import numpy
from numpy import PZERO, euler_gamma  # pylint: disable=no-name-in-module

import ndd.fnsb
from ndd.base import EntropyEstimator, MultiPMFEstimator

logger = logging.getLogger(__name__)

__all__ = ['JSDivergence']


class Plugin(EntropyEstimator):
    """Plugin entropy estimator."""

    def estimator(self, pk, k=None):
        """Estimator definition."""
        k = len(pk)
        if k == 1:
            return PZERO, PZERO
        return ndd.fnsb.plugin(pk, k)


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
        return ndd.fnsb.pseudo(pk, k, self.alpha)


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
        return plugin(pk) + 0.5 * (k - 1) / n


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
        return ndd.fnsb.dirichlet(pk, k, self.alpha)


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
                digamma(delta))


class Grassberger(EntropyEstimator):
    """Grassberger 1988 estimator.

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

        return estimate


class UnderWell(EntropyEstimator):
    """Combination of two estimators.

    Combination of an estimator for the under-sampled regime (asymptotic NSB)
    and another for the well-sampled regime (GRassberger)
    """

    def estimator(self, pk, k=None):
        """Estimator definition."""
        k1 = len(pk > 0)
        n = sum(pk)
        ratio = k1 / n

        under_sampled_estimator = NSBAsymptotic()
        well_sampled_estimator = Grassberger()
        return (ratio**2 * under_sampled_estimator(pk) +
                (1 - ratio**2) * well_sampled_estimator(pk))


class JSDivergence(MultiPMFEstimator):
    """Jensen-Shannon divergence estimator.

    Parameters
    ----------
    entropy_estimator : EntropyEstimator object

    """

    def estimator(self, pk, k=None):
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

        ws = numpy.float64(pk.sum(axis=1))
        ws /= ws.sum()
        if k is None:
            k = pk.shape[1]
        if k == 1:  # single bin
            return PZERO

        return (self.entropy_estimator(pk.sum(axis=0), k) -
                sum(ws[i] * self.entropy_estimator(x, k)
                    for i, x in enumerate(pk)))
