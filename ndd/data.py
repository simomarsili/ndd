# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Contains the DataArray class."""
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
    """Check if x is a sequence or a ndarray."""
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
    """Data container with methods for counts calculation.

    Stores data as a p-by-n numpy ndarray.

    Parameters
    ----------
    ar : array-like, shape (n, p)
        2D array of n samples from p discrete variables.
    axis : int, optional
        The sample-indexing axis. Defaults to axis=0.
    ks : array-like, shape p or int or None, optional
        The alphabet size for the p variables.
        If int: the variables share the same alphabet size.
        If None (default), the alphabet size is unkown.
    k : int or None, optional
        Alphabet size for the p-dimensional joint PMF.

    """

    def __init__(self, ar, axis=0, ks=None, k=None):
        # set data
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
            self._data = ar
        else:
            self._data = ar.data
        self._ks = None
        self._k = k
        if ks is not None:
            self.ks = ks

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
        return '%s(data=\n%s\nks=%s\n)' % (cls.__name__, self.data, self.ks)

    @property
    def data(self):
        """Data are stored as a p-by-n ndarray."""
        return self._data

    @property
    def shape(self):
        """Data shape, p-by-n."""
        return self.data.shape

    @property
    def ks(self):
        """Alphabet size for each of the p variables."""
        return self._ks

    @ks.setter
    def ks(self, value):
        p, _ = self.shape
        if is_sequence(value):
            if len(value) != p:
                raise DataArrayError('len(ks) must be equal to p')
            self._ks = value
        else:
            self._ks = [value] * p

    def counts(self, r=None):
        """Frequency array(s)."""

        def histogram(x):
            _, counts = numpy.unique(x, return_counts=True, axis=1)
            return counts

        if r:
            return (histogram(d) for d in combinations(self, r=r))
        return histogram(self)

    def nunique(self, r=None):
        """
        The product of the number of unique elements observed in the data
        for each variable.

        If r is not None, return a generator for all combinations of r-sized
        sets of variables.
        """
        ns = tuple(len(c) for c in self.counts(r=1))
        if r:
            p, _ = self.shape
            return (numpy.prod([ns[i] for i in idx])
                    for idx in combinations(range(p), r=r))
        return numpy.prod(ns)

    def k(self, r=None):
        """Alphabet size for the joint PMF of variables in dataset.

        If unknown, return the product of the alphabet size of the single
        variables.
        If r is not None, return a generator for all combinations of r-sized
        sets of variables.
        """
        if self._k is not None:
            return self._k

        if self.ks is None:
            return None

        ks = self.ks
        if r:
            p, _ = self.shape
            return (numpy.prod([ks[i] for i in idx])
                    for idx in combinations(range(p), r=r))

        return numpy.prod(ks)

    def iter_data(self, r=None):
        """
        Return tuples of (data, alphabet size) for all r-sized combinations of
        variables.

        If r is None (default): return a tuple (data, alphabet_size).

        """
        cls = type(self)
        if r:
            return zip((cls(c, axis=1) for c in combinations(self, r=r)),
                       self.k(r) or self.nunique(r))
        return self, self.k() or self.nunique()

    def iter_counts(self, r=None):
        """
        Return tuples of (counts, alphabet size) for all r-sized combinations
        of variables.

        If r is None (default): return a tuple (counts, alphabet_size) for the
        data array, where counts are the frequencies of samples in data.

        """
        if r:
            return zip(self.counts(r), self.k(r) or self.nunique(r))
        return self.counts(), self.k() or self.nunique()
