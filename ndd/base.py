# -*- coding: utf-8 -*-
# Author: Simone Marsili
# License: BSD 3 clause
# pylint: disable=c-extension-no-member
"""Classes for computing divergences between distributions."""
import abc
import logging

import numpy
from numpy import PZERO  # pylint: disable=no-name-in-module

from ndd.estimators import EntropyEstimator

__all__ = ['DivergenceEstimator', 'JSDivergence']

logger = logging.getLogger(__name__)


class DivergenceEstimator(EntropyEstimator, abc.ABC):
    """Base class for estimators of divergences."""

    def __init__(self, entropy_estimator):
        super().__init__()
        self._entropy_estimator = entropy_estimator

    @property
    def entropy_estimator(self):
        """EntropyEstimator object."""
        return self._entropy_estimator

    @entropy_estimator.setter
    def entropy_estimator(self, obj):
        """Entropy estimator setter."""
        if isinstance(obj, EntropyEstimator):
            self._entropy_estimator = obj
        else:
            raise TypeError('Not a EntropyEstimator object.')

    @property
    def algorithm(self):
        """Estimator function name."""
        return self.entropy_estimator.__class__.__name__

    @abc.abstractmethod
    def estimator(self, pk, k):
        """Divergence estimator function.

        Parameters
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


class JSDivergence(DivergenceEstimator):
    """Jensen-Shannon divergence estimator.

    Parameters
    ----------
    entropy_estimator : EntropyEstimator object

    """

    def estimator(self, pk, k=None):
        """
        Parameters
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
