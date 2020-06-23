# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
"""Counts class module."""
from itertools import product

import numpy
import pytest

import ndd.counter
import ndd.fnsb
from make_test_ref import SEED

numpy.random.seed(SEED)


class Data:
    def __init__(self, nk=None, zk=None):
        self.nk = nk
        self.zk = zk
        self.counts = None

    def random(self, k=1000, n=100):
        data = numpy.random.randint(k, size=n)
        _, self.counts = numpy.unique(data, return_counts=1)
        self.nk, self.zk = numpy.unique(self.counts, return_counts=1)
        return self

    @staticmethod
    def sorted_are_equal(a, b):
        def int_sort(x):
            return sorted(x.astype(numpy.int32))

        return int_sort(a) == int_sort(b)

    def __eq__(self, other):
        return (self.sorted_are_equal(self.nk, other.nk)
                and self.sorted_are_equal(self.zk, other.zk))


@pytest.fixture(params=product([10, 100, 1000], [10, 100, 1000]))
def data(request):
    return Data().random(*request.param)


def test_fcounter(data):
    fcounter = ndd.fnsb.counter
    fcounter.fit(ar=data.counts)
    assert Data(fcounter.nk, fcounter.zk) == data


def test_unique(data):
    assert Data(*ndd.counter.unique(nk=data.counts, sort=True)) == data
