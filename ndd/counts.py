# -*- coding: utf-8 -*-
"""CountsDistribution class."""
import json
import logging
from collections.abc import Mapping, MappingView
from types import GeneratorType

import numpy

import ndd.fnsb
from ndd.exceptions import NddError

logger = logging.getLogger(__name__)


def unique(nk, sort=True):
    """Return nk, zk"""
    counter = ndd.fnsb.counter
    counter.fit(nk)
    nk = counter.nk
    zk = counter.zk
    unique.counter = counter
    # always return a copy
    if sort:
        ids = numpy.argsort(nk)
        nk = nk[ids]
        zk = zk[ids]
    else:
        nk = numpy.array(nk)
        zk = numpy.array(zk)
    return nk, zk


def as_counts_array(counts):
    """Convert input to counts array."""
    if isinstance(counts, (Mapping, MappingView)):
        return numpy.fromiter(counts.values(), dtype=int)
    if isinstance(counts, (GeneratorType, map, filter)):
        return numpy.fromiter(counts, dtype=int)
    return numpy.asarray(counts)


def check_k(k):
    """
    if k is an integer, just check
    if an array set k = prod(k)
    if None, return

    Raises
    ------
    NddError
        If k is not valid (wrong type, negative, too large...)

    """
    MAX_LOGK = 200 * numpy.log(2)

    if k is None:
        return k
    try:
        k = numpy.float64(k)
    except ValueError:
        raise NddError('%r is not a valid cardinality' % k)
    if k.ndim:
        # if k is a sequence, set k = prod(k)
        if k.ndim > 1:
            raise NddError('k must be a scalar or 1D array')
        logk = numpy.sum(numpy.log(x) for x in k)
        if logk > MAX_LOGK:
            # too large a number; backoff to n_bins?
            # TODO: log warning
            raise NddError('k is too large (%e).'
                           'Must be < 2^200 ' % numpy.exp(logk))
        k = numpy.prod(k)
    else:
        # if a scalar check size
        if k <= 0:
            raise NddError('k must be > 0 (%r)' % k)
        if numpy.log(k) > MAX_LOGK:
            raise NddError('k is too large (%e).' 'Must be < 2^200 ' % k)
    if not k.is_integer():
        raise NddError('k must be a whole number (got %r).' % k)

    return k


class CountsDistribution:
    """
    Contains counts data and statistics.

    Parameters
    ----------
    nk : array-like
        Unique frequencies in a counts array.
    zk : array_like, optional
        Frequencies distribution or "multiplicities".
        Must be len(zk) == len(nk).
    k : int or array-like, optional
        Alphabet size (the number of bins with non-zero probability).
        Must be >= len(nk). A float is a valid input for whole numbers
        (e.g. k=1.e3). If an array, set k = numpy.prod(k).
        Default: k = sum(nk > 0)

    """

    def __init__(self, *, nk=None, zk=None, k=None):
        self.nk = None
        self.k = None
        self.zk = None
        self._n = None
        self._k1 = None
        self.counts = None
        if (nk is None) != (zk is None):
            raise NddError('nk and zk should be passed together.')
        if nk is not None:
            self.nk = as_counts_array(nk)
            self.zk = as_counts_array(zk)
            self._n = numpy.sum(self.zk * self.nk)
            self._k1 = numpy.sum(self.zk[self.nk > 0])
        if k is not None:
            self.k = check_k(k)

    def __repr__(self):
        return 'CountsDistribution(nk=%r, k=%r, zk=%r)' % (self.nk, self.k,
                                                           self.zk)

    def __str__(self):
        return json.dumps(
            {
                'nk': [int(x) for x in self.nk],
                'k': self.k,
                'zk': [int(x) for x in self.zk]
            },
            indent=4)

    def fit(self, counts):
        """Fit nk, zk (multiplicities) from counts array."""
        counts = as_counts_array(counts)
        self.nk, self.zk = unique(counts)
        self._n = numpy.sum(self.zk * self.nk)
        self._k1 = numpy.sum(self.zk[self.nk > 0])
        return self

    @property
    def normalized(self):
        """CountsDistribution are normalized."""
        if self.nk is None:
            return False
        return (len(self.nk) == 1 and self.nk[0] == 0
                and numpy.isclose(sum(self.nk), 1))

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
