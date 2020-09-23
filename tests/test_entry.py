# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
"""Counts class module."""
from ndd.entry import compute_estimate

X = [2, 1, 1, 1, 3]
K = 100


def test_dict():
    data = {k: x for k, x in enumerate(X)}
    assert compute_estimate(X) == compute_estimate(data)


def test_dict_with_alphabet_size():
    data = {'nk': {k: x for k, x in enumerate(X)}, 'k': K}
    assert compute_estimate(X, k=K) == compute_estimate(data)


def test_list_with_alphabet_size():
    data = {'nk': X, 'k': K}
    assert compute_estimate(X, k=K) == compute_estimate(data)
