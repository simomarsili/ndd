# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Functions module."""
import logging

import numpy

import ndd
from ndd.estimators import Entropy, JSDivergence
from ndd.exceptions import (CardinalityError, DataArrayError,
                            EstimatorInputError, HistogramError, NumericError)

__all__ = [
    'entropy', 'jensen_shannon_divergence', 'interaction_information',
    'coinformation', 'mutual_information', 'conditional_entropy', 'histogram',
    'from_data'
]

logger = logging.getLogger(__name__)


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

    Raises
    ------
    NumericError
        If result is NaN

    """

    # pk is an array of counts
    estimator = Entropy(alpha, plugin).fit(pk, k)
    S, err = estimator.estimate_, estimator.err_

    if numpy.isnan(S):
        raise NumericError('NaN value')

    if return_std:
        if err is not None and numpy.isnan(err):
            err = numpy.nan
            logger.warning('nan value for entropy posterior std deviation')
        return S, err

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

    Raises
    ------
    NumericError
        If result is NaN

    """

    estimator = JSDivergence(alpha, plugin).fit(pk, k)
    js = estimator.estimate_

    if numpy.isnan(js):
        raise NumericError('NaN value')

    return js


def _nbins(data):
    """
    The number of unique elements along axis 0. If data is p-dimensional,
    the num. of unique elements for each variable.
    """
    # reshape as a p-by-n array
    return [len(numpy.unique(v)) for v in data]


def histogram(data, axis=1, r=0):
    """Compute an histogram from a data array. Wrapper to numpy.unique.

    Parameters
    ----------
    data : array-like
        A n-by-p array of n samples from p variables.
    axis : int, optional
        The sample-indexing axis
    r : int, optional
        If r > 0, return a generator that yields a bin counts array
        for each possible combination of r variables.

    Returns
    -------
    counts : ndarray
        Bin counts.

    Raises
    ------
    HistogramError
        If r > p.

    """
    from itertools import combinations

    # check data shape
    data = _check_input_data(data)
    if axis == 0:
        data = data.T
    p = data.shape[0]

    if r == 0:
        r = p

    if r > p:
        raise HistogramError(
            'r (%r) is larger than the number of variables (%r)' % (r, p))
    if r < p:
        return (ndd.histogram(d) for d in combinations(data, r=r))

    # statistics for the p-dimensional variable
    _, counts = numpy.unique(data, return_counts=True, axis=1)
    return counts


def from_data(ar, ks=None, axis=1, r=0):
    """
    Given an array of data, return an entropy estimate.

    Paramaters
    ----------
    ar : array-like
        p-by-n array of n samples from p discrete variables.
    ks : 1D p-dimensional array, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis
    r : int, optional
        If r > 0, return a generator yielding estimates for the p-choose-r
        possible combinations of length r from the p variables.

    Returns
    -------
    float
        Entropy estimate

    Raises
    ------
    CardinalityError
        If ks is array-like and len(ks) != p
        If r > 0 and is a scalar.

    """
    from itertools import combinations

    # check data shape
    ar = _check_input_data(ar)
    if axis == 0:
        ar = ar.T
    p = ar.shape[0]
    if r == 0:
        r = p

    # EntropyBasedEstimator objects are callable and return the fitted estimate
    estimator = Entropy()

    if ks is None:
        ks = numpy.array([len(numpy.unique(v)) for v in ar])
    else:
        try:
            ks = numpy.float64(ks)
        except ValueError:
            raise CardinalityError('%s: not a valid cardinality')
        if ks.ndim:
            if len(ks) != p:
                raise CardinalityError('k should have len %s' % p)

    if r != p:
        if ks.ndim == 0:
            raise CardinalityError('For combinations, ks cant be a scalar')

        counts_combinations = histogram(ar, r=r)
        alphabet_size_combinations = (numpy.prod(x)
                                      for x in combinations(ks, r=r))
        return (
            estimator(*args)
            for args in zip(counts_combinations, alphabet_size_combinations))

    counts = histogram(ar)
    return estimator(counts, k=ks)


def interaction_information(ar, ks=None, axis=1, r=0):
    """Interaction information from p-by-n data matrix.

    If p == 2, return an estimate of the mutual information between the
    variables corresponding to the two columns.


    Paramaters
    ----------
    ar : array-like
        p-by-n array of n samples from p discrete variables.
    ks : 1D p-dimensional array, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis
    r : int, optional
        If r > 0, return a generator yielding estimates for the p-choose-r
        possible combinations of length r from the p variables.
        If r == 1, return the entropy for each variable. If r == 2 return the
        mutual information for each possible pair. If r > 2 return the
        interaction information for each possible subset of length r.
        Combinations are ordered as: list(itertools.combinations(range(p), r)).

    Returns
    -------
    float
        Interaction information estimate.

    """
    from itertools import combinations

    # check data shape
    ar = _check_input_data(ar)
    if axis == 0:
        ar = ar.T
    p = ar.shape[0]
    if r == 0:
        r = p

    if ks is None:
        ks = numpy.array([len(numpy.unique(v)) for v in ar])
    else:
        try:
            ks = numpy.float64(ks)
        except ValueError:
            raise CardinalityError('%s: not a valid cardinality')
        if ks.ndim > 0:  # pylint: disable=comparison-with-callable
            if len(ks) != p:
                raise CardinalityError('k should have len %r (%r)' %
                                       (p, len(ks)))
        else:
            raise CardinalityError('ks cant be a scalar')

    def iinfo(X, ks):
        info = 0.0
        px = len(X)
        for ri in range(1, px + 1):
            sgn = (-1)**(px - ri)
            info -= sgn * numpy.sum(from_data(X, ks=ks, r=ri))
        return info

    if r != p:
        data_combinations = combinations(ar, r=r)
        alphabet_size_combinations = (x for x in combinations(ks, r=r))
        return (iinfo(*args)
                for args in zip(data_combinations, alphabet_size_combinations))

    return iinfo(ar, ks)


def coinformation(ar, ks=None, r=0):
    """Coinformation from p-by-n data matrix.

    If p == 2, return an estimate of the mutual information between the
    variables corresponding to the two columns.


    Paramaters
    ----------
    ar : array-like
        p-by-n array of n samples from p discrete variables.
    ks : 1D p-dimensional array, optional
        Alphabet size for each variable.
    r : int, optional
        If r > 0, return a generator yielding estimates for the p-choose-r
        possible combinations of length r from the p variables.
        If r == 1, return the entropy for each variable. If r == 2 return the
        mutual information for each possible pair. If r > 2 return the
        interaction information for each possible subset of length r.
        Combinations are ordered as: list(itertools.combinations(range(p), r)).

    Returns
    -------
    float
        Coinformation estimate.

    """

    # change sign for odd #variables
    return (-1)**ar.shape[0] * interaction_information(ar=ar, ks=ks, r=r)


def mutual_information(ar, ks=None, axis=1):
    """Mutual information from p-by-n data matrix.

    If p > 2, return an estimate of the mutual information for each possible
    pair of variables, ordered as list(itertools.combinations(range(p), r=2)).

    Paramaters
    ----------
    ar : array-like
        p-by-n array of n samples from p discrete variables.
    ks : 1D p-dimensional array, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis

    Returns
    -------
    float
        Coinformation estimate.

    """

    from itertools import combinations

    # check data shape
    ar = _check_input_data(ar)
    if axis == 0:
        ar = ar.T
    p = ar.shape[0]

    if ks is None:
        ks = numpy.array([len(numpy.unique(v)) for v in ar])
    else:
        try:
            ks = numpy.float64(ks)
        except ValueError:
            raise CardinalityError('%s: not a valid cardinality')
        if ks.ndim > 0:  # pylint: disable=comparison-with-callable
            if len(ks) != p:
                raise CardinalityError('k should have len %r (%r)' %
                                       (p, len(ks)))
        else:
            raise CardinalityError('ks cant be a scalar')

    if p > 2:
        h1 = list(from_data(ar, ks=ks, r=1))
        return (h1[i1] + h1[i2] - from_data(ar[[i1, i2]], ks=ks[[i1, i2]])
                for i1, i2 in combinations(range(p), 2))

    return numpy.sum(from_data(ar, ks=ks, r=1)) - from_data(ar, ks=ks)


def conditional_entropy(ar, c, ks=None, axis=1, r=0):
    """
    Coditional entropy estimate from data array.

    Paramaters
    ----------
    ar : array-like
        p-by-n array of n samples from p discrete variables.
    c : int or array-like
        The variables on which entropy is conditioned (as column indices).
    ks : 1D p-dimensional array, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis
    r : int, optional
        If r > 0, return a generator yielding estimates for all possible
        combinations of r variables conditioning on the `c` variables.
        Indices are sorted as:
        list(x for x in collections.combinations(range(p), r=r+len(c))
             if set(c) <= set(x))

    Returns
    -------
    float
        Conditional entropy estimate

    Raises
    ------
    CardinalityError
        If ks is array-like and len(ks) != p
        If r > 0 and is a scalar.

    """
    from itertools import combinations

    # check data shape
    ar = _check_input_data(ar)
    if axis == 0:
        ar = ar.T
    p = ar.shape[0]

    try:
        c = list(c)
    except TypeError:
        c = [c]
    if not set(c) <= set(range(p)):
        return EstimatorInputError('The indices of conditioning variables'
                                   ' are not valid')

    if ks is None:
        ks = numpy.array([len(numpy.unique(v)) for v in ar])
    else:
        try:
            ks = numpy.float64(ks)
        except ValueError:
            raise CardinalityError('%s: not a valid cardinality')
        if ks.ndim:
            if len(ks) != p:
                raise CardinalityError('k should have len %s' % p)

    # EntropyBasedEstimator objects are callable and return the fitted estimate
    estimator = Entropy()

    # Entropy of features on which we are conditioning
    counts = histogram(ar[c])
    hc = estimator(counts, k=ks)

    if r > 0:
        if ks.ndim == 0:
            raise CardinalityError('For combinations, ks cant be a scalar')

        r = r + len(c)
        indices = combinations(range(p), r=r)
        counts_combinations = histogram(ar, r=r)
        alphabet_size_combinations = (numpy.prod(x)
                                      for x in combinations(ks, r=r))
        return (estimator(*args) - hc for ids, *args in zip(
            indices, counts_combinations, alphabet_size_combinations)
                if set(c) <= set(ids))

    counts = histogram(ar)
    return estimator(counts, k=ks) - hc


def _check_input_data(ar):
    # check data shape
    ar = numpy.atleast_2d(ar)
    if ar.ndim != 2:
        raise DataArrayError('input array has %s dimensions; must be 2D' %
                             ar.ndim)
    return ar
