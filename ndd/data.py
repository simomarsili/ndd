# -*- coding: utf-8 -*-
"""Contains DataArray class."""
from collections.abc import Sequence
from itertools import combinations
from operator import itemgetter

import numpy

from ndd.exceptions import CardinalityError, DataArrayError, NddError

# from numbers import Integral


def is_sequence(x):
    """Check if x is a sequence."""
    return (not isinstance(x, str) if isinstance(x, Sequence) else isinstance(
        x, numpy.ndarray))


def is_whole(x):
    """Check if x is a whole number."""
    try:
        x = numpy.float64(x)
    except ValueError:
        return False
    return x.is_integer()


class DataArray(Sequence):
    """Data container."""

    def __init__(self, ar, k=None, axis=0):
        if is_sequence(ar) and isinstance(ar[0], DataArray):
            axis = 1
        _, encoded, counts = numpy.unique(ar,
                                          return_inverse=True,
                                          return_counts=True,
                                          axis=axis)
        encoded.flags['WRITEABLE'] = False
        self.data = encoded
        self.counts = counts
        self._k = None
        self.k = k

    @property
    def nbins(self):
        """Number of observed bins."""
        return len(self.counts)

    @property
    def k(self):
        """Variable cardinality."""
        return self._k

    @property
    def kb(self):
        """Return cardinality if defined else nbins."""
        return self._k or self.nbins

    @k.setter
    def k(self, value):
        if value:
            if not is_whole(value):
                raise CardinalityError('k must be a whole number (got %r)' %
                                       value)
            if value < self.nbins:
                raise DataArrayError('k (%r) must be larger than nbins (%r)' %
                                     (value, self.nbins))
        self._k = numpy.float64(value) if value else None

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.data)


class DataMatrix(Sequence):
    """Data container for multiple DataArray objects."""

    def __init__(self, ar, k=None, axis=1):

        if isinstance(ar, DataArray):
            # a DataArray object
            self.data = (ar, )
            self.shape = 1, len(ar)
        if is_sequence(ar) and isinstance(ar[0], DataArray):
            # a sequence of DataArray objects
            self.data = tuple(x for x in ar)
            self.shape = len(ar), len(ar[0])
        else:
            ar = numpy.atleast_2d(ar)
            if not ar.size:
                raise NddError('Empty data array')
            if ar.ndim > 2:
                raise NddError('Input array has %s dimensions; must be 2D' %
                               ar.ndim)
            if axis == 0:
                ar = ar.T

            self.data = tuple(DataArray(x) for x in ar)
            self.shape = ar.shape

        self.k = k

    def counts(self, r=None):
        """Frequency array(s)."""
        if r:
            return (DataArray(d).counts for d in combinations(self, r=r))
        return DataArray(self).counts

    @property
    def nbins(self):
        """Number of observed bins."""
        return tuple(d.nbins for d in self.data)

    @property
    def k(self):
        """Variable cardinality."""
        return [x.k for x in self]

    @k.setter
    def k(self, value):
        p, _ = self.shape
        if is_sequence(value):
            if len(value) != p:
                raise DataArrayError('len(k) must be equal to p')
        else:
            value = [value] * p
        for x, k in zip(self, value):
            x.k = k

    @property
    def kb(self):
        """Return cardinality if defined else nbins."""
        return [x.k or x.nbins for x in self]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if is_sequence(index):
            return self.__class__(itemgetter(*index)(self))
        return self.data[index]

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.data)
