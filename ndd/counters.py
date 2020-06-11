# -*- coding: utf-8 -*-
"""Counting routines."""
import numbers
from collections import Counter
from collections.abc import Sequence
from types import GeneratorType

import numpy

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
        return not isinstance(obj, Sequence)

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
        counts = Counter()
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
        Replace the Counter dict with a bounter object:
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
        print('sets ', get_indices(sets))
        sets = get_indices(sets)
        source = source[sets]
        return _frequencies_from_array(source)
    return _frequencies_from_array(source)


def multiplicities(source, size_mb=None, sets=None):
    """Return multiplicities array."""

    def get_multi(a):
        a = Counter(a.values())
        return numpy.array(list(a.keys())), numpy.array(list(a.values()))

    counts = frequencies(source, size_mb=size_mb, sets=sets)

    if isinstance(counts, GeneratorType):
        return (get_multi(f) for f in counts)

    return get_multi(counts)
