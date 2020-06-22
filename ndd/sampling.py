# -*- coding: utf-8 -*-
"""Counts class."""
import numpy


class Counts:
    """Statistics from counts"""

    def __init__(self, nk, zk=None):
        if zk is None:  # compute frequency distribution
            self.nk, self.zk = numpy.unique(nk, return_counts=True)
        else:
            self.nk = numpy.asarray(nk)
            self.zk = numpy.asarray(nk)
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
