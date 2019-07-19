# -*- coding: utf-8 -*-
# Author: Simone Marsili
# License: BSD 3 clause
# pylint: disable=c-extension-no-member
"""Compute divergences between distributions."""
import abc
import logging

import numpy
from numpy import PZERO  # pylint: disable=no-name-in-module

import ndd
from ndd.estimators import EntropyEstimator
from ndd.exceptions import NddError

__all__ = ['DivergenceEstimator', 'JSDivergence']

logger = logging.getLogger(__name__)


class DivergenceEstimator(EntropyEstimator, abc.ABC):
    """Base class for estimators of divergences."""

    def __init__(self, entropy='NSB'):
        """Default entropy estimator is NSB."""
        super().__init__()
        self.input_data_ndim = 2
        try:
            self._entropy_estimator = ndd.entropy_estimators[entropy]()
        except KeyError:
            raise NddError(
                'Unknown entropy estimator; valid options are:\n%s' %
                ', '.join(list(ndd.entropy_estimators.keys())))

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
