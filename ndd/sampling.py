# -*- coding: utf-8 -*-
"""Sampling class."""
import numpy


class Sampling:
    """Info on frequency distribution."""

    def __init__(self, nk, zk=None):
        self.nk = numpy.asarray(nk)
        if zk is not None:
            self.zk = numpy.asarray(nk)
        else:
            self.zk = None
        self._n = None
        self._k1 = None

    @property
    def n(self):
        """Number of samples"""
        if self._n is None:
            if self.zk is not None:
                self._n = numpy.sum(self.zk * self.nk)
            else:
                self._n = numpy.sum(self.nk)
        return self._n

    @property
    def k1(self):
        """Number of bins with counts > 0."""
        if self._k1 is None:
            if self.zk is not None:
                self.k1 = numpy.sum(self.zk[self.nk > 0])
            else:
                self.k1 = len(numpy.nonzero(self.nk)[0])
        return self._k1

    @property
    def coincindences(self):
        """Number of coincidences."""
        return self.n - self.k1

    @property
    def sampling_ratio(self):
        """The strongly undersampled regime is defined as ratio < 0.1"""
        return self.coincindences / self.n
