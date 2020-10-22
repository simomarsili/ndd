# -*- coding: utf-8 -*-
"""Basic distribution tests."""
# pylint: disable=missing-docstring
import pytest

import ndd

COUNTS = [4, 12, 4, 5, 3, 1, 5, 1, 2, 2, 2, 2, 11, 3, 4, 12, 12, 1, 2]
K = 100


def rounded(*args, **kwargs):
    result = ndd.entropy(COUNTS, *args, **kwargs)
    return round(result, 3)


def test_base():
    assert rounded() == 2.813


def test_k():
    assert rounded(k=K) == 2.806


def test_plugin():
    assert rounded(estimator='plugin') == 2.635


def test_pmf_plugin():
    assert rounded(estimator='pmf_plugin') == 1.678


def test_miller_madow():
    assert rounded(estimator='miller_madow') == 2.738


def test_wolper_wolf():
    with pytest.raises(TypeError):
        _ = rounded(estimator='wolpert_wolf')


def test_nsb_nok():
    with pytest.raises(ndd.exceptions.NddError):
        _ = rounded(estimator='nsb')


def test_nsb_k():
    assert rounded(estimator='nsb', k=K) == 2.806


def test_asymptotic_nsb():
    assert rounded(estimator='asymptotic_nsb') == 4.612


def test_grassberger():
    assert rounded(estimator='grassberger') == 6.221


def test_auto_estimator_k():
    assert rounded(estimator='auto_estimator', k=K) == 2.806


def test_auto_estimator_nok():
    assert rounded(estimator='auto_estimator') == 2.813
