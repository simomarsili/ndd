# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
"""Test ref results for data with no coincidences."""
import numpy
import pytest
from pytest import approx

from ndd.estimators import AsymptoticNsb, Nsb, Plugin
from ndd.exceptions import NddError

N = (10, 10)
K = (10, 1000)


@pytest.fixture(params=zip(N, K))
def data(request):
    n, k = request.param
    return {'nk': numpy.array([1] * n), 'k': k}


def test_Nsb(data):
    """The Nsb estimate should be somewhat close to log(k)"""
    estimator = Nsb()
    relative_error = 1 - estimator(**data) / numpy.log(data['k'])
    assert 0 < relative_error < 0.2


def test_Asymptotic(data):
    """Should raise an exception"""
    estimator = AsymptoticNsb()
    with pytest.raises(NddError):
        estimator(**data)


def test_Plugin(data):
    """Should be close to the log of #visited bins with frequency > 0"""
    estimator = Plugin(alpha=None)
    k = sum(data['nk'] > 0)
    assert estimator(**data) == approx(numpy.log(k))


def test_Plugin_pseudo(data):
    """Should be close to log(cardinality)"""
    estimator = Plugin(alpha=1)
    assert estimator(**data) == approx(numpy.log(data['k']), rel=1.e-3)
