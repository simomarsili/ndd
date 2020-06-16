# -*- coding: utf-8 -*-
"""Counting routines."""
import collections
import collections.abc
import numbers
from types import GeneratorType

import numpy

from ndd.exceptions import NddError

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


class MultiCounter(collections.abc.MutableMapping):
    """Container for data. Cache multiplicities.

    Parameters
    ----------
    data : array_like
        Data in any form that can be converted to a ndarray. This includes
        lists, lists of tuples, tuples, tuples of tuples, tuples of lists and
        ndarrays.
    stat : str, optional, default: 'multiplicities'
        Valid values are 'counts' and 'multiplicities'
    order : int or None, optional, default: None
        Store statistics up to order `order`. Default: save all statistics.

    """

    def __init__(self, data, stat='multiplicities', order=None):
        if stat in 'counts multiplicities'.split():
            self.stat = stat
        else:
            raise ValueError('Valid values for `star` are '
                             'counts, multiplicities')
        self.statistics = dict()
        self.order = order if order is not None else numpy.float('inf')

        # work with transposed arrays
        self.data = numpy.asarray(data).T

        if self.data.ndim > 2:
            raise ValueError('data should be max 2D')

    def __missing__(self, key):
        raise KeyError(key)

    def __getitem__(self, key):
        try:
            return self.statistics[key]
        except KeyError:
            pass
        return self.__missing__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def counts(self, key=None, k=None):
        """Return counts.
        counts(key) will update the statistics for indices `key`
        if key not in statistics dict.

        Parameters
        ----------
        key : int or tuple or `full`
            Return statistics for the set of features in `key`.
            Defaults: return the statistics for the full set of features.
        k : int or dict or None
            Cardinality. If k is a dict, set k = k[key].
            If `key not in k` and key is a tuple, then set k to the product
            of `(k[x] for x in key)`. No effect if stat='counts'

        Returns
        -------
        keys, values

        """
        if key is None:
            key = 'full'

        if key not in self.statistics:  # compute statistics
            if key == 'full':
                data, order = self.data, 0
            else:
                index, order = self.array_index(key)
                data = self.data[index]

            stats = self._counts(data)
            if order <= self.order:  # save statistics
                self.statistics[key] = stats
        else:
            stats = self.statistics[key]

        keys, values = stats

        if self.stat == 'multiplicities' and k is not None:
            # append statistics for non-observed bins
            if isinstance(k, collections.Mapping):
                try:
                    k = k[key]
                except KeyError:
                    if order > 1:  # use combinatorics
                        try:
                            k = numpy.prod(k[x] for x in key)
                        except KeyError:
                            return NddError('counts(): check k dictionary')
            k = k - sum(values)
            keys.append(0)
            values.append(k)

        return keys, values

    def __len__(self):
        return len(self.statistics)

    def __iter__(self):
        return iter(self.statistics)

    def __contains__(self, key):
        return key in self.statistics

    def __bool__(self):
        return self.data

    def __setitem__(self, key, value):
        """Set item on mapping."""
        self.statistics[key] = value

    def __delitem__(self, key):
        """Delete item from mapping."""
        try:
            del self.statistics[key]
        except KeyError:
            raise KeyError('Key not found in mapping: {!r}'.format(key))

    def popitem(self):
        """Remove and return an item pair from mapping.
        Raise KeyError is data is empty."""
        try:
            return self.statistics.popitem()
        except KeyError:
            raise KeyError('No keys found in mapping.')

    def pop(self, key, *args):  # pylint: disable=arguments-differ
        """Remove *key* from mapping and return its value.
        Raise KeyError if *key* not in mapping."""
        try:
            return self.statistics.pop(key, *args)
        except KeyError:
            raise KeyError('Key not found in mapping: {!r}'.format(key))

    def clear(self):
        'Clear mapping.'
        self.statistics.clear()

    @staticmethod
    def array_index(index):
        """Return int or list."""
        if isinstance(index, numbers.Integral):
            return index, 1
        ids = list(index)
        if len(ids) == 1:
            return ids[0], 1
        return ids, len(ids)

    def _counts(self, data):
        """
        Return a (keys/counts) tuple from data ndarray.
        Samples are indexed by axis -1.

        """
        axis = -1 if self.data.ndim == 2 else None
        u, c = numpy.unique(data, return_counts=1, axis=axis)

        if self.stat == 'counts':
            return list(u), list(c)

        muls = collections.Counter(c)
        keys = list(muls.keys())
        values = list(muls.values())
        keys.append(0)
        values.append(0)
        return keys, values


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
