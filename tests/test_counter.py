# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
"""Counts class module."""
from itertools import product

import numpy
import pytest

from make_test_ref import SEED
from ndd import fnsb
from ndd.counter import Counts, unique

numpy.random.seed(SEED)


@pytest.fixture(params=product([10, 100, 1000], [10, 100, 1000]))
def data(request):
    return Counts().random(*request.param)


def test_fcounter(data):
    fcounter = fnsb.counter
    fcounter.fit(ar=data.counts)
    assert Counts(fcounter.nk, fcounter.zk) == data


def test_unique(data):
    assert Counts(*unique(nk=data.counts, sort=True)) == data
