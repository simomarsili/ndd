# -*- coding: utf-8 -*-
"""Contains DataArray class."""
import numpy

from ndd.exceptions import CardinalityError, DataArrayError


class DataArray(numpy.ndarray):
    """
    Data array helper class.

    Check that input arrays are non-empty 2D arrays.
    """

    #  pylint: disable=access-member-before-definition
    #  pylint: disable=attribute-defined-outside-init

    def __new__(cls, ar, axis):
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
        The number of unique elements along axis 0. If data is p-dimensional,
        the num. of unique elements for each variable.
        """
        if self._ks is None:
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
        elif len(value) != p:
            raise CardinalityError('k should have len %s' % p)

        self._ks = value
