# -*- coding: utf-8 -*-
"""Counts class."""
import numpy

import ndd.fnsb


def unique(nk, sort=False):
    """Return nk, zk"""
    counter = ndd.fnsb.counter
    counter.fit(nk)
    nk = counter.nk
    zk = counter.zk
    unique.counter = counter
    if sort:
        ids = numpy.argsort(nk)
        nk = nk[ids]
        zk = zk[ids]
    return nk, zk


class Counts:
    """Statistics from counts"""

    def __init__(self, nk, zk=None):
        if zk is None:  # compute frequency distribution
            self.nk, self.zk = unique(nk)
            self._n = unique.counter.n_data
            self._k1 = unique.counter.k1
        else:
            self.nk = numpy.asarray(nk)
            self.zk = numpy.asarray(zk)
        self._n = None
        self._k1 = None

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
