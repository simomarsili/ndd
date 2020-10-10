# -*- coding: utf-8 -*-
# Author: Simone Marsili
# License: BSD 3 clause
# pylint: disable=c-extension-no-member
"""Compute divergences between distributions."""
import logging
from abc import ABC, abstractmethod

import numpy
from numpy import PZERO  # pylint: disable=no-name-in-module

from ndd.estimators import EntropyEstimator, Nsb
from ndd.estimators import estimators as entropy_estimators
from ndd.estimators import fit_function
from ndd.exceptions import NddError
from ndd.utils import register_class

__all__ = ['DivergenceEstimator', 'JsDivergence']

logger = logging.getLogger(__name__)

estimators = {}


class DivergenceEstimatorType(type(EntropyEstimator)):
    """Metaclass for entropy estimators."""

    def __new__(cls, name, bases, namespace, **kwargs):
        estimator_class = type.__new__(cls, name, bases, namespace, **kwargs)
        register_class(estimator_class, estimators)
        return estimator_class


class DivergenceEstimator(EntropyEstimator,
                          ABC,
                          metaclass=DivergenceEstimatorType):
    """Base class for estimators of divergences."""

    def __init__(self, entropy=Nsb()):
        """Default entropy estimator is NSB."""
        super(DivergenceEstimator, self).__init__()
        self.input_data_ndim = 2

        estimator_name = type(entropy).__name__
        if estimator_name not in entropy_estimators:
            raise NddError('%s is not a valid entropy estimator' %
                           estimator_name)

        self.entropy_estimator = entropy

    @property
    def algorithm(self):
        """Estimator function name."""
        return self.entropy_estimator.__class__.__name__

    @abstractmethod
    def fit(self, nk, k=None, zk=None):
        """
        Parameters
        ----------
        nk : array_like
            n-by-p array. Different rows correspond to counts from different
            distributions with the same discrete sample space.
        k : int, optional
            Number of bins. k >= p if nk is n-by-p.
            Float values are valid input for whole numbers (e.g. k=1.e3).
            Defaults to nk.shape[1].

        Returns
        -------
        self : object
            Returns the instance itself.

        Raises
        ------
        CountsError
            If nk is not a 2D array.

        """


class JsDivergence(DivergenceEstimator):
    """Jensen-Shannon divergence estimator.

    Parameters
    ----------
    entropy_estimator : EntropyEstimator object

    """

    @fit_function
    def fit(self, nk, k=None, zk=None):
        ws = numpy.float64(nk.sum(axis=1))
        ws /= ws.sum()
        if k is None:
            k = nk.shape[1]
        if k == 1:  # single bin
            return PZERO

        self.estimate_ = (self.entropy_estimator(nk.sum(axis=0), k=k) -
                          sum(ws[i] * self.entropy_estimator(x, k=k)
                              for i, x in enumerate(nk)))
        return self
