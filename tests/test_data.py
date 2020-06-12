# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
"""Test DataArray class"""
import numpy

from ndd.data import DataArray

DATA = numpy.array([[1, 1, 1, 1], [1, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]])

DARRAY = DataArray(DATA)


def test_data():
    data = DataArray(DATA)
    assert (data.data == DATA.T).all()


def test_data_array():
    assert DataArray(DARRAY).data is DARRAY.data


def test_axis():
    data = DataArray(DATA, axis=1)
    assert (data.data == DATA).all()


def test_nunique():
    assert DARRAY.nunique() == numpy.prod([1, 2, 2, 2])


def test_counts():
    # check the most frequent symbol for each column
    assert [numpy.amax(c) for c in DARRAY.counts(1)] == [4, 2, 2, 3]


def test_iter_data():
    assert next(DARRAY.iter_data(1))[0] == DataArray(DATA[:, 0])


def test_iter_counts():
    assert (list(DARRAY.iter_counts(1))[0] == list(
        DataArray(DATA[:, 0]).iter_counts(1))[0])


if __name__ == '__main__':
    test_data_array()
