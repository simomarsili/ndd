# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Base classes module."""
import logging

import numpy

from ndd.base import EntropyBasedEstimator
from ndd.exceptions import CountsError

logger = logging.getLogger(__name__)

__all__ = ['Entropy', 'JSDivergence']


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
