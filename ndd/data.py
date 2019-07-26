# -*- coding: utf-8 -*-
"""Contains DataArray class."""
from itertools import combinations

import numpy

from ndd.exceptions import CardinalityError, DataArrayError

# from numbers import Integral


class DataArray(numpy.ndarray):
    """
    Data array helper class.

    Check that input arrays are non-empty 2D arrays.
    """

    #  pylint: disable=access-member-before-definition
    #  pylint: disable=attribute-defined-outside-init
    #  pylint: disable=protected-access

    def __new__(cls, ar, axis=1):
        if isinstance(ar, cls):
            return ar

        ar = numpy.atleast_2d(ar)

        if not ar.size:
            raise DataArrayError('Empty data array')

        if ar.ndim > 2:
            raise DataArrayError('Input array has %s dimensions; must be 2D' %
                                 ar.ndim)
        if axis == 0:
            ar = ar.T

        ar.flags['WRITEABLE'] = False

        return ar.view(cls)

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        default_attributes = {'_ks': None}
        self.__dict__.update(default_attributes)

    @property
    def ks(self):
        """
        Alphabet size for each variable.
        """
        if self._ks is None:
            if self.ndim == 1:
                self._ks = len(numpy.unique(self))
            else:
                self._ks = numpy.array([len(numpy.unique(v)) for v in self])
        return self._ks

    @ks.setter
    def ks(self, value):
        """
        Set ks values.
        """
        try:
            value = numpy.float64(value)
        except ValueError:
            raise CardinalityError('%s: not a valid cardinality')

        p = self.shape[0]  # pylint: disable=unsubscriptable-object
        if value.ndim == 0:
            value = numpy.ones(p) * value

        if numpy.all(value >= self.ks):
            self._ks = value
        else:

            raise CardinalityError('ks cannot be set')

    def __getitem__(self, index):
        # support slicing for ks attribute
        ar = super().__getitem__(index)
        if isinstance(ar, DataArray):
            if isinstance(index, tuple):
                index = index[0]
            if self._ks is None:
                ar._ks = None
            else:
                ar._ks = self._ks[index]
        return ar

    def combinations(self, r):
        """Data from combinations of different sets of variables."""
        return combinations(self, r)
