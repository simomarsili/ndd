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

EPS = 0.0001
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


@pytest.mark.parametrize('setting, kwargs, result', CASES)
def test_entropy(setting, kwargs, result):
    """Basic tests."""
    counts = random_counts(**setting)
    test_result = ndd.entropy(counts, k=setting['k'], **kwargs)
    assert numpy.abs(test_result - numpy.float64(result)) < EPS


def test_histogram_ndarray():
    N, P = 100, 2
    data = random_ndarray(N, P, SEED)
    assert ndd.entropy(ndd.histogram(data), ndd.nsb.n_unique(data)) == 6.412863794582687
