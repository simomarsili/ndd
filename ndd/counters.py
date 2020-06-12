# -*- coding: utf-8 -*-
"""Counting routines."""
import collections
import collections.abc
import numbers
from abc import ABC, abstractmethod
from types import GeneratorType

import numpy

from ndd.base import BaseEstimator

try:
    from pandas import DataFrame, Series
except ImportError:
    pandas_is_installed = False
else:
    pandas_is_installed = True

try:
    from bounter import bounter
except ImportError:
    bounter_is_installed = False
else:
    bounter_is_installed = True


class BaseCounter(BaseEstimator, ABC):
    """Fit frequencies/multiplicities from discrete data."""

    def __init__(self, columns=None):
        self.counts_ = None
        self.multiplicities_ = None

        multiple_sets = (columns
                         and (not isinstance(columns,
                                             (tuple, numbers.Integral))))
        if columns is not None:
            if multiple_sets:
                columns = (self.get_indices(c) for c in columns)
            else:
                columns = self.get_indices(columns)
        self.multiple_sets = multiple_sets
        self.columns = columns

    @abstractmethod
    def get_indices(self, index):
        """Preprocess data."""

    @abstractmethod
    def preprocess(self, data):
        """Preprocess data."""

    @abstractmethod
    def fit(self, data):
        """Fit to data."""


class ArrayCounter(BaseCounter):
    """Counts from array data."""

    def get_indices(self, index):
        """Return int or list."""
        if isinstance(index, numbers.Integral):
            return index
        if isinstance(index, tuple):
            ids = list(index)
        if len(ids) == 1:
            ids = ids[0]
        return ids

    def preprocess(self, data):
        """Prepare array"""

        if pandas_is_installed:
            if isinstance(data, (DataFrame, Series)):
                data = data.to_numpy()

        if data.ndim > 2:
            raise ValueError('data should be max 2D')

        if isinstance(data, numpy.ndarray):
            # work with transposed arrays
            data = data.T
        else:
            data = numpy.asarray(data).T

        if self.columns is not None and data.ndim == 2:
            if self.multiple_sets:
                print(list(self.columns))
                return (data[s] for s in self.columns)
            return data[self.columns]
        return data

    @staticmethod
    def counts(data):
        """Return a {key: counts} dict."""
        axis = -1 if data.ndim == 2 else None
        if axis:
            u, c = numpy.unique(data, axis=axis, return_counts=1)
            return {k: n for k, n in zip(zip(*u), c)}
        u, c = numpy.unique(data, return_counts=1)
        return {k: n for k, n in zip(u, c)}

    def fit(self, data):
        """Return a {key: counts} dict."""
        data = self.preprocess(data)
        if self.multiple_sets:
            self.counts_ = (self.counts(d) for d in data)
        else:
            self.counts_ = self.counts(data)


def _frequencies_from_array(ar):
    """Return a {key: counts} dict."""
    axis = -1 if ar.ndim == 2 else None
    if axis:
        u, c = numpy.unique(ar, axis=axis, return_counts=1)
        return {k: n for k, n in zip(zip(*u), c)}
    u, c = numpy.unique(ar, return_counts=1)
    return {k: n for k, n in zip(u, c)}


def _frequencies_from_records(records, ids=None, size_mb=None):
    """Frequencies from records generator."""

    if ids is not None:
        x = set()
        if isinstance(ids, numbers.Integral):
            x.add(ids)
        else:
            x.update(ids)
        ids = x

    is_1d = None

    def is_sequence(obj):
        if isinstance(obj, str):
            return True
        return not isinstance(obj, collections.abc.Sequence)

    def stringify(features):
        nonlocal is_1d
        if is_1d is None:
            is_1d = is_sequence(features)

        if is_1d:
            return str(features)

        if ids:  # select set of indices
            return ' '.join(str(x) for j, x in enumerate(features) if j in ids)
        return ' '.join(str(x) for x in features)

    if callable(records):
        records = records()

    if size_mb:
        # approximate counting using bounter
        counts = bounter(size_mb=size_mb)
    else:
        counts = collections.Counter()
    counts.update(stringify(row) for row in records)
    return counts


def frequencies(source, size_mb=None, sets=None):  # pylint: disable=too-many-branches
    """
    Compute frequencies.

    Parameters
    ----------
    source : generator function/dataframe/numpy 2D array/list of lists
        The generator function returns an iterator over records.
        If dataframe/2D array: 2D array of n samples from p discrete variables.
    size_mb : int
        Limit the size of counts dict using approximate counting.
        Replace the collections.Counter dict with a bounter object:
        https://github.com/RaRe-Technologies/bounter
    sets : int or tuple or iterable
        m-tuple of indices. Return the frequencies of the classes for the
        m-dimensional variables defined by the set of indices.
        If an iterable of tuples, return a generator with the frequencies for
        each set in `sets`.

    Returns
    -------
    counts : dict-like
        A dict-like object.
        **Supports iteration over key-count pairs as `counts.items()`**

    """

    if size_mb and not bounter_is_installed:
        # ignore size_mb
        size_mb = None

    source_is_generator = (isinstance(source, GeneratorType)
                           or callable(source))

    def get_indices(ids):
        """Return int or list."""
        if isinstance(ids, numbers.Integral):
            return ids
        if isinstance(ids, tuple):
            ids = list(ids)
        if len(ids) == 1:
            ids = ids[0]
        return ids

    if sets is not None:
        multiple_sets = not isinstance(sets, (tuple, numbers.Integral))
        if multiple_sets:
            sets = (get_indices(s) for s in sets)
        else:
            sets = get_indices(sets)

    if source_is_generator:
        if sets is not None:
            if multiple_sets:
                if not callable(source):
                    raise ValueError('Cant run multiples sets '
                                     'if source is a generator')
                return (_frequencies_from_records(source,
                                                  ids=get_indices(s),
                                                  size_mb=size_mb)
                        for s in sets)
            return _frequencies_from_records(source,
                                             ids=get_indices(sets),
                                             size_mb=size_mb)
        return _frequencies_from_records(source, size_mb=size_mb)

    if pandas_is_installed:
        if isinstance(source, (DataFrame, Series)):
            source = source.to_numpy()

    if source.ndim > 2:
        raise ValueError('source should be max 2D')

    if isinstance(source, numpy.ndarray):
        # work with transposed arrays
        source = numpy.transpose(source)
    else:
        source = numpy.asarray(source).T

    # print(source)

    if sets is not None and source.ndim == 2:  # ignore sets for 1D arrays
        if multiple_sets:
            return (_frequencies_from_array(source[get_indices(s)])
                    for s in sets)
        sets = get_indices(sets)
        source = source[sets]
        return _frequencies_from_array(source)
    return _frequencies_from_array(source)


def multiplicities(source, size_mb=None, sets=None):
    """Return multiplicities array."""

    def get_multi(a):
        a = collections.Counter(a.values())
        return numpy.array(list(a.keys())), numpy.array(list(a.values()))

    counts = frequencies(source, size_mb=size_mb, sets=sets)

    if isinstance(counts, GeneratorType):
        return (get_multi(f) for f in counts)

    return get_multi(counts)
