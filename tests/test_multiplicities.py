# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, redefined-outer-name
"""Tests for estimators working with multiplicities and related routines"""
from collections import Counter

import numpy
import pytest

from make_test_ref import SEED, approx
from ndd import fnsb
from ndd.counters import MultiCounter
from ndd.estimators import NSB, Grassberger, MillerMadow, Plugin

K = 4
N = 10000
P = 3


@pytest.fixture
def data_1d():
    numpy.random.seed(SEED)
    return numpy.random.randint(K, size=N)


@pytest.fixture
def data_2d():
    numpy.random.seed(SEED)
    return numpy.random.randint(K, size=(N, P))


@pytest.fixture
def counts_1d(data_1d):
    counter = MultiCounter(data_1d, stat='counts')
    return counter.counts()[1]


@pytest.fixture
def multi_1d(data_1d):
    counter = MultiCounter(data_1d, stat='multiplicities')
    return counter.counts(k=K)


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


def identical_sorted(a, b):
    return sorted(list(a)) == sorted(list(b))


def test_nsb_from_multiplicities(data_1d):
    frequencies = compute_frequencies(data_1d)
    hn, hz = compute_multiplicities(data_1d)
    estimate_from_counts = fnsb.nsb(frequencies, K)[0]
    estimate_from_multiplicities = fnsb.nsb_from_multiplicities(hn, hz, K)[0]
    assert estimate_from_multiplicities == approx(estimate_from_counts)


def test_counter_1d_counts(data_1d):
    freqs0 = compute_frequencies(data_1d)
    counter = MultiCounter(data_1d, stat='counts')
    freqs = counter.counts()[1]
    assert identical_sorted(freqs, freqs0)


def test_counter_2d(data_2d):
    freqs0 = numpy.unique(data_2d, return_counts=1, axis=0)[1]
    mult0 = Counter(freqs0)
    mult0[0] = 0
    counter = MultiCounter(data_2d)
    mult = counter.counts()[1]
    assert identical_sorted(mult, mult0.values())


def test_counter_2d_columns(data_2d):
    ids = (1, 2)
    freqs0 = numpy.unique(data_2d[:, list(ids)], return_counts=1, axis=0)[1]
    mult0 = Counter(freqs0)
    mult0[0] = 0
    counter = MultiCounter(data_2d)
    mult = counter.counts(ids)[1]
    assert identical_sorted(mult, mult0.values())


def test_nsb(counts_1d, multi_1d):
    estimate_from_counts = fnsb.nsb(counts_1d, K)[0]
    estimate_from_multiplicities = fnsb.nsb_from_multiplicities(
        multi_1d[0], multi_1d[1], K)[0]
    assert estimate_from_multiplicities == approx(estimate_from_counts)


def test_nsb_estimator(counts_1d, multi_1d):
    estimate_from_counts = NSB()(counts_1d, k=K)
    nk, zk = multi_1d
    estimate_from_multiplicities = NSB()(nk, zk=zk, k=K)
    assert estimate_from_multiplicities == approx(estimate_from_counts)


def test_ww(counts_1d, multi_1d):
    alpha = 0.1
    estimate_from_counts = fnsb.ww(counts_1d, K, alpha)[0]
    estimate_from_multiplicities = fnsb.ww_from_multiplicities(
        multi_1d[0], multi_1d[1], K, alpha)[0]
    assert estimate_from_multiplicities == approx(estimate_from_counts)


def test_ww_estimator(counts_1d, multi_1d):
    alpha = 0.1
    estimate_from_counts = NSB(alpha=alpha)(counts_1d, k=K)
    nk, zk = multi_1d
    estimate_from_multiplicities = NSB(alpha=alpha)(nk, zk=zk, k=K)
    assert estimate_from_multiplicities == approx(estimate_from_counts)


def test_grassberger_estimator(counts_1d, multi_1d):
    est = Grassberger()
    estimate_from_counts = est(counts_1d, k=K)
    nk, zk = multi_1d
    estimate_from_multiplicities = est(nk, zk=zk, k=K)
    assert estimate_from_multiplicities == approx(estimate_from_counts)


def test_plugin_estimator(counts_1d, multi_1d):
    est = Plugin()
    estimate_from_counts = est(counts_1d, k=K)
    nk, zk = multi_1d
    estimate_from_multiplicities = est(nk, zk=zk, k=K)
    assert estimate_from_multiplicities == approx(estimate_from_counts)


def test_millermadow_estimator(counts_1d, multi_1d):
    est = MillerMadow()
    estimate_from_counts = est(counts_1d, k=K)
    nk, zk = multi_1d
    estimate_from_multiplicities = est(nk, zk=zk, k=K)
    assert estimate_from_multiplicities == approx(estimate_from_counts)
