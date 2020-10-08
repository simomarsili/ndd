# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
"""Tests for the entropy() function."""
import numpy
import pytest

import ndd
from inputs import Counts, Pmf
from make_test_ref import approx


@pytest.fixture
def pmf():
    return Pmf()


@pytest.fixture
def pmf_with_zeros():
    return Pmf(zero=0.5)


@pytest.fixture
def counts():
    return Counts()


def test_pmf(pmf):
    ref = pmf.entropy
    assert ndd.entropy(pmf.pk) == approx(ref)


def test_pmf_with_zeros(pmf_with_zeros):
    ref = pmf_with_zeros.entropy
    print(pmf_with_zeros.pk.sum())
    assert ndd.entropy(pmf_with_zeros.pk) == approx(ref)


def test_counts(counts):
    estimator = ndd.estimators.AutoEstimator()
    assert ndd.entropy(counts.nk) == approx(
        counts.entropy(estimator=estimator))


def test_unnormalized_pmf():
    counts = numpy.random.random(size=100)  # pylint: disable=no-member
    pk = counts / counts.sum()
    assert ndd.entropy(counts) == approx(Pmf().entropy_from_pmf(pk))
