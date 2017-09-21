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

@pytest.fixture()
def cases():
    import json
    with open('data.json', 'r') as _jf:
        return json.load(_jf)
    
@pytest.mark.parametrize('a, ns, nd, result', cases())
def test_basic(a, ns, nd, result):
    """Basic tests."""
    np.random.seed(123)
    pp=np.random.dirichlet([a]*ns)
    data = np.random.multinomial(nd,pp)
    assert ndd.entropy(data) == np.float64(result)
