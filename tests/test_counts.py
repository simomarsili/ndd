# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
"""Counts class module."""
from itertools import product

import numpy
import pytest

from make_test_ref import SEED
from ndd import fnsb
from ndd.counts import Counts, unique

numpy.random.seed(SEED)

P = 100
X = numpy.random.randint(2, size=P)


@pytest.fixture(params=product([10, 100, 1000], [10, 100, 1000]))
def data(request):
    return Counts().random(*request.param)


def test_fcounter(data):
    fcounter = fnsb.counter
    fcounter.fit(ar=data.counts)
    assert Counts(nk=fcounter.nk, zk=fcounter.zk) == data


def test_unique(data):
    nk, zk = unique(nk=data.counts, sort=True)
    assert Counts(nk=nk, zk=zk) == data


def test_mapping():
    y = dict(zip(range(P), X))
    assert Counts(nk=X) == Counts(nk=y)


def test_generator():
    y = (n for n in X)
    assert Counts(nk=X) == Counts(nk=y)


def test_series():
    try:
        from pandas import Series
    except ImportError:
        assert 1
    else:
        y = Series(X)
        assert Counts(nk=X) == Counts(nk=y)
