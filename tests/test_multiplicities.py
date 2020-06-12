# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
"""Tests for estimators working with multiplicities and related routines"""
from collections import Counter

import numpy

import ndd.fnsb

K = 10000
N = 1000
data = numpy.random.randint(K, size=N)


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


frequencies = compute_frequencies(data)

hn, hz = compute_multiplicities(data)


def test_nsb_from_multiplicities():
    estimate_from_counts = ndd.fnsb.nsb(frequencies, K)[0]
    estimate_from_multiplicities = ndd.fnsb.nsb(frequencies, K)[0]
    assert numpy.isclose(estimate_from_multiplicities, estimate_from_counts)
