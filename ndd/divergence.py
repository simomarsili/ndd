# -*- coding: utf-8 -*-
# Author: Simone Marsili
# License: BSD 3 clause
# pylint: disable=c-extension-no-member
"""Compute divergences between distributions."""
import logging
from abc import ABCMeta, abstractmethod

import numpy
from numpy import PZERO  # pylint: disable=no-name-in-module

import ndd
from ndd.estimators import NSB, EntropyEstimator, check_input
from ndd.exceptions import NddError

__all__ = ['DivergenceEstimator', 'JSDivergence']

logger = logging.getLogger(__name__)

# compatible with both Python 2 and 3
# https://stackoverflow.com/a/38668373
ABC = ABCMeta('ABC', (object, ), {'__slots__': ()})


class DivergenceEstimator(EntropyEstimator, ABC):
    """Base class for estimators of divergences."""

    def __init__(self, entropy=NSB()):
        """Default entropy estimator is NSB."""
        super(DivergenceEstimator, self).__init__()
        self.input_data_ndim = 2

        estimator_name = type(entropy).__name__
        if estimator_name not in ndd.entropy_estimators:
            raise NddError('%s is not a valid entropy estimator' %
                           estimator_name)

        self.entropy_estimator = entropy

    @property
    def algorithm(self):
        """Estimator function name."""
        return self.entropy_estimator.__class__.__name__

    @abstractmethod
    def fit(self, pk, k=None):
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


class JSDivergence(DivergenceEstimator):
    """Jensen-Shannon divergence estimator.

    Parameters
    ----------
    entropy_estimator : EntropyEstimator object

    """

    @check_input
    def fit(self, pk, k=None):
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

        self.estimate_ = (self.entropy_estimator(pk.sum(axis=0), k=k) -
                          sum(ws[i] * self.entropy_estimator(x, k=k)
                              for i, x in enumerate(pk)))
        return self
