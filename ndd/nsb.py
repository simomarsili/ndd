# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""Functions module."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next,
    oct, open, pow, round, super, filter, map, zip)
import numpy
import ndd
from ndd.estimators import Entropy, JSDivergence

__copyright__ = "Copyright (C) 2016,2017 Simone Marsili"
__license__ = "BSD 3 clause"
__author__ = "Simone Marsili (simomarsili@gmail.com)"
__all__ = ['entropy',
           'jensen_shannon_divergence',
           'histogram', ]


def entropy(pk, k=None, alpha=None, plugin=False, return_std=False):
    """
    Return a Bayesian estimate of the entropy from an array of counts.

    Return a Bayesian estimate of the entropy of an unknown discrete
    distribution from an input array of counts pk.

    Parameters
    ----------

    pk : array-like
        The number of occurrences of a set of bins.
    k : int or array-like, optional
        Total number of bins (taking into account unobserved bins);
        k >= len(pk). A float is a valid input for whole numbers (e.g. k=1.e3).
        If an array, set k = numpy.prod(k).
        Defaults to n_bins.
    alpha : float, optional
        If alpha is not None, use a single Dirichlet prior with concentration
        parameter alpha (fixed alpha estimator). alpha > 0.0.
    plugin : boolean, optional
        If True, return a 'plugin' estimate of the entropy. The discrete
        distribution is estimated from the empirical frequencies over bins
        and inserted into the entropy definition (plugin estimator).
        If alpha is passed in combination with plugin=True, add
        alpha pseudocounts to each frequency count (pseudocount estimator).
    return_std : boolean, optional
        If True, also return an approximated value for the standard deviation
        over the entropy posterior.

    Returns
    -------
    entropy : float
        Entropy estimate.
    std : float, optional
        Uncertainty in the entropy estimate
        (approximated standard deviation over the entropy posterior).
        Only if `return_std` is True.

    """

    # pk is an array of counts
    estimator = Entropy(alpha, plugin).fit(pk, k)
    S, err = estimator.estimate_, estimator.err_

    if numpy.isnan(S) or (err is not None and numpy.isnan(err)):
        raise FloatingPointError("NaN value")

    if return_std:
        return S, err
    else:
        return S


def jensen_shannon_divergence(pk, k=None, alpha=None, plugin=False):
    """
    Return the Jensen-Shannon divergence from a matrix of counts.

    Return an estimate of the Jensen-Shannon divergence between
    n_distributions unknown discrete distributions from a
    n_distributions-by-n_bins input array of counts.
    The estimate (in nats) is computed as a combination of single Bayesian
    entropy estimates. If the total number of samples varies among the
    distributions, the function returns the divergence between the
    distributions with weights proportional to the total number of samples in
    each row (see the general definition of Jensen-Shannon divergence:
    https://en.wikipedia.org/wiki/Jensen-Shannon_divergence).

    Parameters
    ----------

    pk : array-like, shape (n_distributions, n_bins)
        Matrix of frequency counts. Each row corresponds to the number of
        occurrences of a set of bins from a different distribution.
    k : int or array-like, optional
        Total number of bins (taking into account unobserved bins); k >= n_bins
        A float is a valid input for whole numbers (e.g. k=1.e3).
        If an array, set k = numpy.prod(k).
        Defaults to n_bins.
    alpha : float, optional
        If not None, the entropy estimator uses a single Dirichlet prior with
        concentration parameter alpha (fixed alpha estimator). alpha > 0.0.
    plugin : boolean, optional
        If True, use a 'plugin' estimator for the entropy.
        If alpha is passed in combination with plugin == True, add alpha
        pseudoconts to the frequency counts in the plugin estimate.

    Returns
    -------
    float
        Jensen-Shannon divergence.

    """

    estimator = JSDivergence(alpha, plugin).fit(pk, k)
    js = estimator.estimate_

    if numpy.isnan(js):
        raise FloatingPointError("NaN value")

    return js


def _nbins(data):
    """
    The number of unique elements along axis 0. If data is p-dimensional,
    the num. of unique elements for each variable.
    """
    # reshape as a p-by-n array
    data = ndd.nsb.as_data_array(data)
    return [len(numpy.unique(v)) for v in data]


def histogram(data, axis=0, r=0):
    """Compute an histogram from data. Wrapper to numpy.unique.

    Parameters
    ----------
    data : array-like
        An array of n samples from p variables.
    axis : int, optional
        The sample-indexing axis
    r : int, optional
        If r > 0, return a generator that yields bin counts for each possible
        combination of r variables.

    Returns
    -------
    counts : ndarray
        Bin counts.

    """
    from itertools import combinations
    # reshape as a p-by-n array
    data = ndd.nsb.as_data_array(data, axis=axis)
    p = data.shape[0]
    if r > p:
        raise ValueError(
            'r (%r) is larger than the number of variables (%r)' % (r, p))
    if r == 0:
        # statistics for the p-dimensional variable
        _, counts = numpy.unique(data, return_counts=True, axis=1)
        return counts
    else:
        return (ndd.histogram(d, axis=1) for d in combinations(data, r=r))


def as_data_array(ar, axis=0):
    """
    For a 2D n-by-p data array, transpose it.
    For a generic ndarray, flatten the subarrays indexed by axis 0
    axis : int, optional
        The sample-indexing axis
    """

    ar = numpy.asanyarray(ar)

    if ar.ndim == 1:
        n = ar.shape[0]
        ar = ar.reshape(1, n)
    elif ar.ndim == 2:
        if axis == 0:
            ar = ar.T
    elif ar.ndim > 2:
        if axis != 0:
            try:
                ar = numpy.swapaxes(ar, axis, 0)
            except ValueError:
                raise numpy.AxisError(axis, ar.ndim)
        n = ar.shape[0]
        ar = ar.reshape(n, -1)
        ar = ar.T

    return numpy.ascontiguousarray(ar)


def _from_data(ar, ks=None, axis=0, r=0):
    """
    Given an array of data, return an entropy estimate.

    Paramaters
    ----------
    ar : array-like
        n-by-p array of n samples from p discrete variables.
    ks : 1D p-dimensional array, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis.
    r : int, optional
        If r > 0, return a generator yielding estimates for the p-choose-r
        possible combinations of length r from the op variables.

    Returns
    -------
    float
        Entropy estimate

    """
    from itertools import combinations

    ar = as_data_array(ar, axis=axis)
    p = ar.shape[0]

    if ks is None:
        ks = numpy.array([len(numpy.unique(v)) for v in ar])
    else:
        try:
            if len(ks) == p:
                ks = numpy.array(ks)
            else:
                raise ValueError("k should have len %s" % p)
        except TypeError:
            raise

    entropy_estimator = Entropy()
    if r == 0:
        counts = histogram(ar, axis=1)
        k = numpy.prod(ks)
        return entropy_estimator(counts, k=k)
    else:
        counts_combinations = histogram(ar, axis=1, r=r)
        alphabet_size_combinations = (numpy.prod(x)
                                      for x in combinations(ks, r=r))
        return (
            entropy_estimator(c, k=k)
            for c, k in zip(counts_combinations, alphabet_size_combinations))
