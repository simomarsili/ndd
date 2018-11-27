# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)
import ndd
import pytest
import numpy
from numpy import random as random

EPS = 0.0001

def random_counts(n=None, k=None, alpha=None):
    random.seed(123)
    pp = random.dirichlet([alpha]*k)
    return random.multinomial(n, pp)

import json
with open('data.json', 'r') as _jf:
    CASES = json.load(_jf)
    
@pytest.mark.parametrize('setting, kwargs, result', CASES['NSB'])
def test_NSB(setting, kwargs, result):
    """Basic tests."""
    counts = random_counts(**setting)
    assert numpy.abs(ndd.entropy(counts, **kwargs) - numpy.float64(result)) < EPS

@pytest.mark.parametrize('setting, kwargs, result', CASES['Dirichlet'])
def test_Dirichlet(setting, kwargs, result):
    """Basic tests."""
    counts = random_counts(**setting)
    assert numpy.abs(ndd.entropy(counts, **kwargs) - numpy.float64(result)) < EPS

@pytest.mark.parametrize('setting, kwargs, result', CASES['ML'])
def test_ML(setting, kwargs, result):
    """Basic tests."""
    counts = random_counts(**setting)
    assert numpy.abs(ndd.entropy(counts, **kwargs) - numpy.float64(result)) < EPS

@pytest.mark.parametrize('setting, kwargs, result', CASES['pseudo'])
def test_pseudo(setting, kwargs, result):
    """Basic tests."""
    counts = random_counts(**setting)
    assert numpy.abs(ndd.entropy(counts, **kwargs) - numpy.float64(result)) < EPS
