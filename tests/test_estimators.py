# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
"""Estimators tests."""
import pytest

import ndd
from inputs import Pmf
from make_test_ref import approx


@pytest.fixture
def pmf():
    return Pmf()


def test_PmfPlugin(pmf):
    """Test estimator from PMF."""
    estimator = ndd.estimators.PmfPlugin()
    ref = pmf.entropy
    assert estimator(pmf.pk) == approx(ref)
