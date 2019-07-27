# -*- coding: utf-8 -*-
"""Contains DataArray class."""
from collections.abc import Sequence
from itertools import combinations

import numpy

from ndd.exceptions import CardinalityError, DataArrayError, NddError

# from numbers import Integral


class Data1D(Sequence):
    """Data container."""

    def __init__(self, ar, k=None, axis=0):
        _, encoded, counts = numpy.unique(ar,
                                          return_inverse=True,
                                          return_counts=True,
                                          axis=axis)
        encoded.flags['WRITEABLE'] = False
        self.data = encoded
        self.nbins = len(counts)
        self.counts = counts

        if k is not None:
            if k < self.nbins:
                raise ValueError('k must be larger than nbins')
            self.k = k

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.data)


class DataMatrix(Sequence):
    """Data container for multiple Data1D objects."""

    def __init__(self, ar, k=None, axis=0):
        try:
            k = numpy.float64(k)
        except ValueError:
            raise CardinalityError('%s: not a valid cardinality')

        ar = numpy.atleast_2d(ar)
        if not ar.size:
            raise NddError('Empty data array')
        if ar.ndim > 2:
            raise NddError('Input array has %s dimensions; must be 2D' %
                           ar.ndim)
        if axis == 0:
            ar = ar.T

        p, _ = ar.shape

        if k.ndim == 0:
            self.k = numpy.ones(p) * k
        else:
            if len(k) != p:
                raise ValueError('len(k) must be equal to p')
            self.k = k

        self.data = [Data1D(x, k=k) for x, k in zip(ar, self.k)]
        self.nbins = [d.nbins for d in self.data]
        self.counts = [d.counts for d in self.data]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.data)


class DataArray(numpy.ndarray):
    """
    Data array helper class.

    Check that input arrays are non-empty 2D arrays.
    """

    #  pylint: disable=access-member-before-definition
    #  pylint: disable=attribute-defined-outside-init
    #  pylint: disable=protected-access

    def __new__(cls, ar, axis=1):
        if isinstance(ar, cls):
            return ar

        ar = numpy.atleast_2d(ar)

        if not ar.size:
            raise DataArrayError('Empty data array')

        if ar.ndim > 2:
            raise DataArrayError('Input array has %s dimensions; must be 2D' %
                                 ar.ndim)
        if axis == 0:
            ar = ar.T

        ar.flags['WRITEABLE'] = False

        return ar.view(cls)

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        default_attributes = {'_ks': None}
        self.__dict__.update(default_attributes)

    @property
    def ks(self):
        """
        Alphabet size for each variable.
        """
        if self._ks is None:
            if self.ndim == 1:
                self._ks = len(numpy.unique(self))
            else:
                self._ks = numpy.array([len(numpy.unique(v)) for v in self])
        return self._ks

    @ks.setter
    def ks(self, value):
        """
        Set ks values.
        """
        try:
            value = numpy.float64(value)
        except ValueError:
            raise CardinalityError('%s: not a valid cardinality')

        p = self.shape[0]  # pylint: disable=unsubscriptable-object
        if value.ndim == 0:
            value = numpy.ones(p) * value

        if numpy.all(value >= self.ks):
            self._ks = value
        else:

            raise CardinalityError('ks cannot be set')

    def __getitem__(self, index):
        # support slicing for ks attribute
        ar = super().__getitem__(index)
        if isinstance(ar, DataArray):
            if isinstance(index, tuple):
                index = index[0]
            if self._ks is None:
                ar._ks = None
            else:
                ar._ks = self._ks[index]
        return ar

    def combinations(self, r):
        """Data from combinations of different sets of variables."""
        return combinations(self, r)
