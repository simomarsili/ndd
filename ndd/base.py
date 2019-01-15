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
    """Base class for estimators from sklearn.

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
    """Mixin class for EntropyEstimator.

    Contains methods to select an estimator and compute an estimates from data.
    """

    def plugin_estimator(self, pk, k):
        return ndd.fnsb.plugin(pk, k), None

    def pseudocounts_estimator(self, pk, k, alpha):
        return ndd.fnsb.pseudo(pk, k, alpha), None

    def ww_estimator(self, pk, k, alpha):
        return ndd.fnsb.dirichlet(pk, k, alpha), None

    def nsb_estimator(self, pk, k):
        return ndd.fnsb.nsb(pk, k)

    def select_estimator(self):
        """
        Return an estimator function for the object.
        Possible estimators are:
        - NSB (Nemenman-Shafee-Bialek)
        - WW (Wolper-Wolf)
        - "plugin"
        - pseudocounts-regularized plugin
        """

        if self.plugin:
            if self.alpha is None:
                return self.plugin_estimator
            else:
                def pseudocounts_estimator(pk, k):
                    return self.pseudocounts_estimator(pk, k, self.alpha)
                return pseudocounts_estimator
        else:
            if self.alpha is None:
                return self.nsb_estimator
            else:
                def ww_estimator(pk, k):
                    return self.ww_estimator(pk, k, self.alpha)
                return ww_estimator

    def estimator(self, pk, k):
        """
        Return an entropy estimate from counts and the size of sample space.

        Parameters
        ----------
        pk : array-like
            An array of non-negative integers (counts array).
        k  : int or sequence
            Size of the sample space.
            Float values are valid input for whole numbers (e.g. k=1.e3).
            If a sequence, set k = numpy.prod(k).

        Returns
        -------
        estimate : float
            Entropy estimate
        err : float or None
            A measure of uncertainty in the estimate. None if not available.

        """
        pk = self.check_pk(pk)
        if k is None:
            k = len(pk)
        k = self.check_k(k)

        return self.estimator_function(pk, k)

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
    """Extend the BaseEstimator to estimators of entropy-derived quantities.

    Specific estimators should extend the EntropyEstimator class with a fit()
    method. The fit() method must set the estimator object attributes
    estimate and err (using the estimator_function method).

    Parameters
    ----------
    alpha : float, optional
        If not None: Wolpert-Wolf estimator function (fixed alpha).
        A single Dirichlet prior with concentration parameter alpha.
        alpha > 0.0.
    plugin : boolean, optional
        If True: 'plugin' estimator function.
        The discrete distribution is estimated from the empirical frequencies
        over bins and inserted into the entropy definition (plugin estimator).
        If alpha is passed in combination with plugin=True, add
        alpha pseudocounts to each frequency count (pseudocount estimator).

    Attributes
    ----------
    estimate_ : float
        Entropy estimate
    err_ : float or None
        A measure of uncertainty in the estimate. None if not available.

    """

    def __init__(self, alpha=None, plugin=False):
        self.alpha = self.check_alpha(alpha)
        self.plugin = plugin
        self._estimator_function = None
        self._algorithm = None

        self.estimate_ = None
        self.err_ = None

    def __call__(self, *args, **kwargs):
        """Fit and return the estimated value."""
        return self.fit(*args, **kwargs).estimate_

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

    @property
    def estimator_function(self):
        """Entropy estimator function."""
        if self._estimator_function is None:
            self._estimator_function = self.select_estimator()
        return self._estimator_function

    @property
    def algorithm(self):
        """Estimator function name."""
        return self.estimator_function.__name__.split('_')[0]

    def fit(self):
        """Set the estimated parameters."""
        raise NotImplemented


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
