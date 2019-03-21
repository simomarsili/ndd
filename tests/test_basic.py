# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)
import os
import json
import pytest
import numpy
from numpy import random as random
import ndd

SEED = 123


def tests_dir():
    """Return None is no tests dir."""
    cwd = os.getcwd()
    basename = os.path.basename(cwd)
    if basename == 'tests':
        return cwd
    else:
        tests_dir = os.path.join(cwd, 'tests')
        if os.path.exists(tests_dir):
            return tests_dir


def random_counts(n=None, k=None, alpha=None):
    random.seed(123)
    pp = random.dirichlet([alpha]*k)
    return random.multinomial(n, pp)


def random_ndarray(n, p, seed):
    import string
    random.seed(seed)
    alphabet = list(string.ascii_uppercase)
    return random.choice(alphabet, size=(n, p))


def random_tuple_generator(n, p, seed):
    import string
    random.seed(seed)
    alphabet = list(string.ascii_uppercase)
    for j in range(n):
        yield tuple(random.choice(alphabet, size=p))


with open(os.path.join(tests_dir(), 'data.json'), 'r') as _jf:
    CASES = json.load(_jf)


@pytest.mark.parametrize('setting, kwargs, ref_result', CASES)
def test_entropy(setting, kwargs, ref_result):
    """Basic tests."""
    counts = random_counts(**setting)
    test_result = ndd.entropy(counts, k=setting['k'], **kwargs)
    assert numpy.isclose(test_result, ref_result)


def test_histogram_ndarray():
    N, P = 100, 3
    data = random_ndarray(N, P, SEED)
    ref_result = 9.107550241712808
    assert numpy.isclose(ndd.entropy(ndd.histogram(data),
                                     k=ndd.nsb._nbins(data)), ref_result)


def test_from_data():
    N, P = 100, 3
    data = random_ndarray(N, P, SEED)
    ref_result = 9.107550241712808
    assert numpy.isclose(ndd.nsb._from_data(data, ks=ndd.nsb._nbins(data)),
                         ref_result)


def test_combinations_from_data():
    N, P = 100, 3
    data = random_ndarray(N, P, SEED)
    hs_pairs = ndd.nsb._from_data(data, ks=ndd.nsb._nbins(data), r=2)
    ref_result = 18.84820751635297
    assert numpy.isclose(numpy.sum(hs_pairs), ref_result)


def test_KLD():
    ALPHA, N, P = 1.0, 100, 20
    random.seed(SEED)
    qk = random.dirichlet([ALPHA]*P)
    pk = random.multinomial(N, qk)
    estimator = ndd.estimators.KLDivergence()
    ref_result = -0.04299973796573253
    assert numpy.isclose(estimator(pk, qk), ref_result)


def test_JSD():
    ALPHA, N, P = 1.0, 100, 20
    random.seed(SEED)
    pk = random.dirichlet([ALPHA]*P)
    counts = random.multinomial(N, pk, size=4)
    estimator = ndd.estimators.JSDivergence()
    ref_result = -0.01804523405829217
    assert numpy.isclose(estimator(counts), ref_result)
