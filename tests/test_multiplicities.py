# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
"""Tests for estimators working with multiplicities and related routines"""
from collections import Counter
from itertools import combinations

import numpy
import pytest

import ndd.fnsb
from ndd.counters import ArrayCounter

K = 10
N = 10000
P = 4
data_1d = numpy.random.randint(K, size=N)
data_2d = numpy.random.randint(K, size=(N, P))


def compute_frequencies(a):
    """Frequencies from 1D array"""
    return list(Counter(a).values())


def compute_multiplicities(a):
    """Return a tuple (frequencies, multiplicities) from 1D array"""
    freqs = compute_frequencies(a)
    counter = Counter(freqs)
    freqs, mults = (list(x) for x in zip(*counter.items()))
    # add unobserved bins
    n_observed_bins = sum(mults)
    freqs.append(0)
    mults.append(K - n_observed_bins)
    return freqs, mults


def identical_lists(a, b):
    a = list(a)
    b = list(b)
    return all([i == j for i, j in zip(a, b)])


def identical_sorted_lists(a, b):
    a = sorted(list(a))
    b = sorted(list(b))
    return identical_lists(a, b)


def test_nsb_from_multiplicities():
    frequencies = compute_frequencies(data_1d)
    hn, hz = compute_multiplicities(data_1d)
    estimate_from_counts = ndd.fnsb.nsb(frequencies, K)[0]
    estimate_from_multiplicities = ndd.fnsb.nsb_from_multiplicities(hn, hz)[0]
    assert numpy.isclose(estimate_from_multiplicities, estimate_from_counts)


def test_arraycounter_1d():
    freqs0 = sorted(compute_frequencies(data_1d))
    counter = ArrayCounter()
    counter.fit(data_1d)
    counts = counter.counts_
    freqs = sorted(list(counts.values()))
    assert str(freqs) == str(freqs0)


def test_arraycounter_2d():
    freqs0 = numpy.unique(data_2d, return_counts=1, axis=0)[1]
    counter = ArrayCounter()
    counter.fit(data_2d)
    counts = counter.counts_
    freqs = counts.values()
    assert identical_sorted_lists(freqs, freqs0)


def test_arraycounter_2d_columns():
    ids = (1, 2)
    freqs0 = numpy.unique(data_2d[:, list(ids)], return_counts=1, axis=0)[1]
    counter = ArrayCounter(columns=ids)
    counter.fit(data_2d)
    counts = counter.counts_
    freqs = counts.values()
    assert identical_sorted_lists(freqs, freqs0)


@pytest.mark.parametrize('set_size', range(1, P - 1))
def test_arraycounter_2d_sets(set_size):
    def sets(n):
        return combinations(range(P), n)

    freqs0 = []
    for s in sets(set_size):
        f0 = numpy.unique(data_2d[:, list(s)], return_counts=1, axis=0)[1]
        freqs0 += list(f0)
    print(freqs0)
    counter = ArrayCounter(columns=sets(set_size))
    counter.fit(data_2d)
    freqs = []
    for c in counter.counts_:
        freqs += list(c.values())
    assert identical_sorted_lists(freqs, freqs0)
