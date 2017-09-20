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
import numpy as np

tests = [
    (0.01, 100, 100, 0.21622160227928622),
    (0.1, 100, 100, 2.569415088680637),
    (1.0, 100, 100, 4.322717746281727)]

@pytest.mark.parametrize('a, ns, nd, result', tests)
def test_basic(a, ns, nd, result):
    """Basic tests."""
    np.random.seed(123)
    pp=np.random.dirichlet([a]*ns)
    data = np.random.multinomial(nd,pp)
    assert ndd.entropy(data) == np.float64(result)
