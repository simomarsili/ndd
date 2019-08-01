# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""DataArray class."""
import logging
from itertools import combinations
from numbers import Integral
from operator import itemgetter

import numpy

from ndd.exceptions import DataArrayError, NddError

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

__all__ = ['DataArray']

logger = logging.getLogger(__name__)


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
    """Data container with methods for counts calculation."""

    def __init__(self, ar, axis=0, k=None):
        if not isinstance(ar, self.__class__):
            ar = numpy.atleast_2d(ar)
            if not ar.size:
                raise NddError('Empty data array')
            if ar.ndim > 2:
                raise NddError('Input array has %s dimensions; must be 2D' %
                               ar.ndim)
            if ar.shape[0] > 1:
                # take 1D array as single variable, n samples data
                if ar.shape[0] > 1 and axis == 0:
                    ar = ar.T
            self.data = ar
        self._k = None
        if k is not None:
            self.k = k

    def __getitem__(self, index):
        cls = type(self)
        if isinstance(index, slice):
            return cls(self.data[index], axis=1)
        if isinstance(index, Integral):
            return self.data[index]
        if is_sequence(index):
            return cls(itemgetter(*index)(self), axis=1)
        raise NddError('%s is not avalid index type' % type(index))

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        cls = type(self)
        return '%s(data=\n%s\nk=%s\n)' % (cls.__name__, self.data, self.k)

    @property
    def shape(self):
        """Data shape, p-by-n."""
        return self.data.shape

    def counts(self, r=None):
        """Frequency array(s)."""

        def histogram(x):
            _, counts = numpy.unique(x, return_counts=True, axis=1)
            return counts

        if r:
            return (histogram(d) for d in combinations(self, r=r))
        return histogram(self)

    def nbins(self, r=None):
        """#bins."""
        ns = tuple(len(c) for c in self.counts(r=1))
        if r:
            p, _ = self.shape
            return (numpy.prod([ns[i] for i in idx])
                    for idx in combinations(range(p), r=r))
        return numpy.prod(ns)

    @property
    def k(self):
        """Data cardinality (passed to Data instance)."""
        return self._k

    @k.setter
    def k(self, value):
        p, _ = self.shape
        if is_sequence(value):
            if len(value) != p:
                raise DataArrayError('len(k) must be equal to p')
            self._k = value
        else:
            self._k = [value] * p

    def ks(self, r=None):
        """#bins."""
        if self.k is None:
            return None
        ns = self.k
        if r:
            p, _ = self.shape
            return (numpy.prod([ns[i] for i in idx])
                    for idx in combinations(range(p), r=r))
        return numpy.prod(ns)

    def iter_data(self, r=None):
        """
        Return tuples of (data, cardinality) over r-sized sets of variables.

        If cardinality is unknown, defaults to nbins for single variables
        or to the product of nbins for sets of variables.
        """
        cls = type(self)
        if r:
            return zip((cls(c, axis=1) for c in combinations(self, r=r)),
                       self.ks(r) or self.nbins(r))
        return self, self.ks() or self.nbins()

    def iter_counts(self, r=None):
        """
        Return tuples of (counts, cardinality) over r-sized sets of variables.

        If cardinality is unknown, defaults to nbins for single variables
        or to the product of nbins for sets of variables.
        """
        if r:
            return zip(self.counts(r), self.ks(r) or self.nbins(r))
        return self.counts(), self.ks() or self.nbins()
