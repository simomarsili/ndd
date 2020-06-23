# -*- coding: utf-8 -*-
"""Counts class."""
import numpy

import ndd.fnsb


def unique(nk, sort=True):
    """Return nk, zk"""
    counter = ndd.fnsb.counter
    counter.fit(nk)
    # return a copy
    nk = counter.nk
    zk = counter.zk
    unique.counter = counter
    if sort:
        ids = numpy.argsort(nk)
        nk = nk[ids]
        zk = zk[ids]
    else:
        nk = numpy.array(nk)
        zk = numpy.array(zk)
    return nk, zk


class Counts:
    """Statistics from counts"""

    def __init__(self, nk=None, zk=None):
        self.nk = None
        self.zk = None
        self._n = None
        self._k1 = None
        self.counts = None
        if nk is not None:
            self.nk = numpy.asarray(nk)
            if zk is None:
                self.counts = self.nk
                self.fit(self.nk)
        if zk is not None:
            self.zk = numpy.asarray(zk)

    def fit(self, counts):
        """Fit nk, zk (multiplicities) data."""
        self.nk, self.zk = unique(counts)
        self._n = unique.counter.n_data
        self._k1 = unique.counter.k1

    def random(self, k=1000, n=100):
        """Generate random counts and fit multiplicities."""
        a = numpy.random.randint(k, size=n)
        _, self.counts = numpy.unique(a, return_counts=1)
        self.nk, self.zk = numpy.unique(self.counts, return_counts=1)
        return self

    @staticmethod
    def sorted_are_equal(a, b):
        """True if sorted arrays are equal."""

        def int_sort(x):
            return sorted(x.astype(numpy.int32))

        return int_sort(a) == int_sort(b)

    def __eq__(self, other):
        return (self.sorted_are_equal(self.nk, other.nk)
                and self.sorted_are_equal(self.zk, other.zk))

    @property
    def n(self):
        """Number of samples"""
        if self._n is None:
            self._n = numpy.sum(self.zk * self.nk)
        return self._n

    @property
    def k1(self):
        """Number of bins with counts > 0."""
        if self._k1 is None:
            self._k1 = numpy.sum(self.zk[self.nk > 0])
        return self._k1

    @property
    def coincidences(self):
        """Number of coincidences."""
        return self.n - self.k1

    @property
    def sampling_ratio(self):
        """The strongly undersampled regime is defined as ratio < 0.1"""
        return self.coincidences / self.n

    @property
    def multiplicities(self):
        """Return counts and their frequencies as (counts, frequencies)."""
        return self.nk, self.zk
