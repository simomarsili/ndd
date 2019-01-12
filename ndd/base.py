# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""Base classes module."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next,
    oct, open, pow, round, super, filter, map, zip)
import numpy
try:
    from inspect import signature  # pylint: disable=wrong-import-order
except ImportError:
    from ndd.funcsigs import signature
from collections import defaultdict  # pylint: disable=wrong-import-order
import ndd.fnsb


class BaseEstimator(object):
    """Base class for entropy estimators.
    Methods from sklearn BaseEstimator (only).

    The class is consistent with sklearn estimator API:

    All estimator objects expose a ``fit`` method that takes a dataset
    (usually a 2-d array):

    >>> estimator.fit(data)

    **Estimator parameters**: All the parameters of an estimator can be set
    when it is instantiated or by modifying the corresponding attribute::

    >>> estimator = Estimator(param1=1, param2=2)
    >>> estimator.param1
    1

    **Estimated parameters**: When data is fitted with an estimator,
    parameters are estimated from the data at hand. All the estimated
    parameters are attributes of the estimator object ending by an
    underscore::

    >>> estimator.estimated_param_ #doctest: +SKIP
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != 'self' and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError(
                    'Invalid parameter %s for estimator %s. '
                    'Check the list of available parameters '
                    'with `estimator.get_params().keys()`.' % (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (
            class_name,
            _pprint(
                self.get_params(deep=False),
                offset=len(class_name),
            ),
        )

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)


class EntropyEstimatorMixin(object):
    """Estimator method (interface to Fortran routines).
    """

    def _plugin_estimator(self, pk, k):
        return ndd.fnsb.plugin(pk, k), None

    def _pseudocounts_estimator(self, pk, k, alpha):
        return ndd.fnsb.pseudo(pk, k, alpha), None

    def _ww_estimator(self, pk, k, alpha):
        return ndd.fnsb.dirichlet(pk, k, alpha), None

    def _nsb_estimator(self, pk, k):
        return ndd.fnsb.nsb(pk, k)

    def estimator(self, pk, k):
        pk = self.check_pk(pk)
        if k is None:
            k = len(pk)
        k = self.check_k(k)
        if self._estimator is None:
            if self.plugin:
                if self.alpha is None:
                    self._estimator = self._plugin_estimator
                else:
                    self._estimator = lambda pk, k: self._pseudocounts_estimator(pk, k, self.alpha)  # pylint: disable=redefined-variable-type
            else:
                if self.alpha is None:
                    self._estimator = self._nsb_estimator
                else:
                    self._estimator = lambda pk, k: self._ww_estimator(pk, k, self.alpha)
        return self._estimator(pk, k)

    @staticmethod
    def check_pk(a):
        a = numpy.float64(a).flatten()
        not_integers = not numpy.all([x.is_integer() for x in a])
        negative = numpy.any([a < 0])
        if not_integers:
            raise ValueError('counts array has non-integer values')
        if negative:
            raise ValueError('counts array has negative values')
        return numpy.int32(a)

    @staticmethod
    def check_k(k):
        """
        if k is None, set k = number of bins
        if k is an integer, just check
        ik an array set k = prod(k)
        """
        MAX_LOGK = 150 * numpy.log(2)

        try:
            k = numpy.float64(k)
        except ValueError:
            raise
        if k.ndim:
            # if k is a sequence, set k = prod(k)
            if k.ndim > 1:
                raise ValueError('k must be a scalar or 1D array')
            logk = numpy.sum(numpy.log(x) for x in k)
            if logk > MAX_LOGK:
                # too large a number; backoff to n_bins?
                # TODO: log warning
                raise ValueError('k (%r) larger than %r' %
                                 (numpy.exp(logk), numpy.exp(MAX_LOGK)))
            else:
                k = numpy.prod(k)
        else:
            # if a scalar check size
            if numpy.log(k) > MAX_LOGK:
                raise ValueError(
                    'k (%r) larger than %r' % (k, numpy.exp(MAX_LOGK)))
        if not k.is_integer():
            raise ValueError("k (%s) should be a whole number." % k)
        return k


class EntropyEstimator(BaseEstimator, EntropyEstimatorMixin):
    """Specialize to estimates of entropy-derived quantities."""

    def __init__(self, alpha=None, plugin=False):
        self.alpha = self.check_alpha(alpha)
        self.plugin = plugin

        self.estimate = None
        self.std = None
        self._estimator = None

    def __call__(self, *args, **kwargs):
        """Fit and return the estimated value."""
        return self.fit(*args, **kwargs).estimate

    def check_alpha(self, a):
        if a is None:
            return a
        try:
            a = numpy.float64(a)
        except ValueError:
            raise ValueError('alpha (%r) should be numeric.' % a)
        if a < 0:
            raise ValueError('Negative alpha value: %r' % a)
        return a


def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'
    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int
        The offset in characters to add at the begin of each line.
    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr
    """
    # Do a multi-line justified repr:
    options = numpy.get_printoptions()
    numpy.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if isinstance(v, float):
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    numpy.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines
