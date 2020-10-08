# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
"""Tests for the entropy() function."""
import numpy

from make_test_ref import SEED


class Pmf:
    """PMF class.

    Parameters
    ----------
    alpha : float
        Concentration parameter.
    k : int
        Alphabet size.
    zero : float or None
        Fraction of bins with exactly zero probability.

    """

    def __init__(self, alpha=0.1, k=10000, zero=0):
        numpy.random.seed(SEED)
        self.alpha = alpha
        self.k = k
        self.zero = zero
        self._pk = self._generate_pk(self.alpha, self.k, self.zero)
        self._entropy = None

    @property
    def pk(self):
        return self._pk

    @staticmethod
    def _generate_pk(alpha, k, zero=0):
        """Return a Dirichlet sample."""
        pk = numpy.random.dirichlet([alpha] * k)
        if zero:
            n_zero = numpy.random.binomial(k, zero)
            pk[:n_zero] = 0
            pk /= pk.sum()
            pk = pk[n_zero:]
        return pk

    def randomize(self):
        """Reset pk to a random pmf."""
        self._pk = self._generate_pk(self.alpha, self.k, self.zero)
        self._entropy = None
        return self

    @staticmethod
    def entropy_from_pmf(a):
        pk = numpy.asarray(a)
        pk = pk[pk > 0]
        return -numpy.sum(pk * numpy.log(pk))

    @property
    def entropy(self):
        """Entropy for PMF"""
        if self._entropy is None:
            self._entropy = self.entropy_from_pmf(self.pk)
        return self._entropy


class Counts:
    def __init__(self, n=100, pmf=None, **kwargs):
        """
        Counts class.

        Parameters
        ----------
        n : int
            Number of samples.
        pmf : Pmf object, optional
            Alphabet size.

        """
        numpy.random.seed(SEED)
        self.n = n

        if pmf is not None:
            self.pmf = pmf
        else:
            self.pmf = Pmf(**kwargs)

        self._nk = self._generate_nk(self.n, self.pmf.pk)
        self._entropy = None

    @property
    def nk(self):
        return self._nk

    @staticmethod
    def _generate_nk(n, pk):
        """Return a Multinomial sample."""
        return numpy.random.multinomial(n, pk)

    def randomize(self):
        self.nk = self._generate_nk(self.n, self.pmf.pk)
        self._entropy = None
        return self

    @staticmethod
    def entropy_from_counts(a, estimator, **kwargs):
        nk = numpy.asarray(a)
        estimator.fit(nk, **kwargs)
        return estimator.estimate_

    def entropy(self, estimator, **kwargs):
        """Entropy estimate from counts using `estimator`."""
        if self._entropy is None:
            self._entropy = self.entropy_from_counts(self.nk, estimator,
                                                     **kwargs)
        return self._entropy
