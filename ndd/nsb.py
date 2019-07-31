# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""
Functions for entropy and information measures estimation.
"""
import logging

import numpy

from ndd.estimators import Entropy, JSDivergence
from ndd.exceptions import (CardinalityError, CombinationError, DataArrayError,
                            EstimatorInputError, PmfError)

__all__ = [
    'entropy',
    'from_data',
    'jensen_shannon_divergence',
    'kullback_leibler_divergence',
    'interaction_information',
    'coinformation',
    'mutual_information',
    'conditional_entropy',
    'histogram',
]

logger = logging.getLogger(__name__)


def entropy(pk, k=None, alpha=None, plugin=False, return_std=False):
    """
    Entropy estimate from an array of counts.

    Return a Bayesian estimate for the entropy of an unknown discrete
    distribution from an input array of counts pk.

    Parameters
    ----------
    pk : array-like
        The number of occurrences of a set of bins.
    k : int or array-like, optional
        Total number of bins (including unobserved bins); k >= len(pk).
        A float is a valid input for whole numbers (e.g. k=1.e3).
        If an array, set k = numpy.prod(k). Defaults to len(pk).
    alpha : float, optional
        If not None: Wolpert-Wolf entropy estimator (fixed alpha).
        Use a single Dirichlet prior with concentration parameter alpha.
        alpha > 0.0.
    plugin : boolean, optional
        If True, return a 'plugin' estimate of the entropy. The discrete
        distribution is estimated from the empirical frequencies over bins
        and inserted into the entropy definition (plugin estimator).
        If alpha is passed in combination with plugin=True, add
        alpha pseudocounts to each frequency count (pseudocount estimator).
    return_std : boolean, optional
        If True, also return an approximation for the standard deviation
        over the entropy posterior.

    Returns
    -------
    entropy : float
        Entropy estimate.
    std : float, optional
        Uncertainty in the entropy estimate. Only if `return_std` is True.

    """

    # pk is an array of counts
    estimator = Entropy(alpha, plugin).fit(pk, k)
    S, err = estimator.estimate_, estimator.err_

    if numpy.isnan(S):
        logger.warning('nan value for entropy estimate')
        S = numpy.nan

    if return_std:
        if err is not None and numpy.isnan(err):
            err = numpy.nan
            logger.warning('nan value for entropy posterior std deviation')
        return S, err

    return S


def from_data(ar, ks=None, axis=1, r=None):
    """
    Entropy estimate from data matrix.

    Paramaters
    ----------
    ar : array-like, shape (p, n)
        2D array of n samples from p discrete variables.
    ks : int or 1D array of length p, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis. Defaults to 1.
    r : int, optional; ; 1<=r<=p.
        If passed, return a generator yielding estimates for the p-choose-r
        possible combinations of r variables.

    Returns
    -------
    float
        Entropy estimate

    """
    from itertools import combinations

    # check data shape
    ar = _check_data(ar, axis)

    # EntropyBasedEstimator objects are callable and return the fitted estimate
    estimator = Entropy()

    ks = _check_ks(ks, ar)

    if r is not None:
        if ks.ndim == 0:
            raise CardinalityError('For combinations, ks cant be a scalar')
        r = _check_r(r, ar)

        counts_combinations = histogram(ar, r=r)
        alphabet_size_combinations = (numpy.prod(x)
                                      for x in combinations(ks, r=r))
        return (
            estimator(*args)
            for args in zip(counts_combinations, alphabet_size_combinations))

    counts = histogram(ar)
    return estimator(counts, k=ks)


def jensen_shannon_divergence(pk, k=None, alpha=None, plugin=False):
    """
    Return the Jensen-Shannon divergence from a m-by-p matrix of counts.

    Return an estimate of the Jensen-Shannon divergence between
    m unknown discrete distributions from a m-by-p input array of counts.
    The estimate (in nats) is computed as a combination of single Bayesian
    entropy estimates. If the total number of samples varies among the
    distributions, the function returns a weighted divergence with weights
    proportional to the total number of samples in each row
    (see the general definition of Jensen-Shannon divergence:
    https://en.wikipedia.org/wiki/Jensen-Shannon_divergence).

    Parameters
    ----------

    pk : array-like, shape (m, p)
        Matrix of frequency counts. Each row corresponds to the number of
        occurrences of a set of bins from a different distribution.
    k : int or array-like, optional
        Total number of bins (including unobserved bins); k >= p.
        A float is a valid input for whole numbers (e.g. k=1.e3).
        If an array, set k = numpy.prod(k). Defaults to p.
    alpha : float, optional
        If not None: Wolpert-Wolf entropy estimator (fixed alpha).
        Use a single Dirichlet prior with concentration parameter alpha.
        alpha > 0.0.
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
        logger.warning('nan value for JS divergence')
        js = numpy.nan

    return js


def kullback_leibler_divergence(pk, qk, k=None, alpha=None, plugin=False):
    """
    Kullback-Leibler divergence given counts pk and a reference PMF qk.

    Return an estimate of the Kullback-Leibler given an array of counts pk and
    a reference PMF qk. The estimate (in nats) is computed as:
    - S_p - sum(pk * log(qk)) / sum(pk)
    where S_p is the entropy estimate from counts pk.

    Parameters
    ----------
    pk : array_like
        The number of occurrences of a set of bins.
    qk : array_like
        Reference PMF in sum(pk log(pk/qk). len(qk) = len(pk).
        Must be a valid PMF (non-negative, normalized).
    k : int or array-like, optional
        Total number of bins (including unobserved bins); k >= p.
        A float is a valid input for whole numbers (e.g. k=1.e3).
        If an array, set k = numpy.prod(k). Defaults to len(pk).
    alpha : float, optional
        If not None: Wolpert-Wolf entropy estimator (fixed alpha).
        Use a single Dirichlet prior with concentration parameter alpha.
        alpha > 0.0.
    plugin : boolean, optional
        If True, use a 'plugin' estimator for the entropy.
        If alpha is passed in combination with plugin == True, add alpha
        pseudoconts to the frequency counts in the plugin estimate.

    Returns
    -------
    float
        Kullback-Leibler divergence.

    """

    if is_pmf(qk):
        log_qk = numpy.log(qk)
    else:
        raise PmfError('qk must be a valid PMF')

    if len(log_qk) != len(pk):
        raise PmfError('qk and pk must have the same length.')

    if k == 1:  # single bin
        kl = 0.0
    else:
        estimator = Entropy(alpha, plugin).fit(pk, k)
        kl = -estimator.estimate_ - numpy.sum(pk * log_qk) / float(sum(pk))
        if numpy.isnan(kl):
            logger.warning('nan value for KL divergence')
            kl = numpy.nan

    return kl


def interaction_information(ar, ks=None, axis=1, r=None):
    """Interaction information from data matrix.

    See Eq.10 in:
    Timme, Nicholas, et al.
    "Synergy, redundancy, and multivariate information measures:
    an experimentalist's perspective."
    Journal of computational neuroscience 36.2 (2014): 119-140.

    Paramaters
    ----------
    ar : array-like
        p-by-n array of n samples from p discrete variables.
    ks : 1D array of length p, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis
    r : int, optional; 1<=r<=p.
        If passed, return a generator yielding estimates for the p-choose-r
        possible combinations of r variables.
        Combinations are ordered as: list(itertools.combinations(range(p), r)).

    Returns
    -------
    float
        Interaction information estimate.

    Raises
    ------
    CardinalityError
        If len(ks) != p.

    """
    from itertools import combinations

    # check data shape
    ar = _check_data(ar, axis)

    ks = _check_ks(ks, ar)
    if ks.ndim == 0:
        raise CardinalityError('ks cant be a scalar')

    if r is not None:
        r = _check_r(r, ar)

        data_combinations = combinations(ar, r=r)
        alphabet_size_combinations = (x for x in combinations(ks, r=r))
        return (iinfo(*args)
                for args in zip(data_combinations, alphabet_size_combinations))

    return iinfo(ar, ks)


def coinformation(ar, ks=None, r=None):
    """Coinformation from data matrix.

    See Eq.11 in:
    Timme, Nicholas, et al.
    "Synergy, redundancy, and multivariate information measures:
    an experimentalist's perspective."
    Journal of computational neuroscience 36.2 (2014): 119-140.

    The coinformation reduces to the entropy for a single variable and to the
    mutual information for a pair of variables.

    Paramaters
    ----------
    ar : array-like
        p-by-n array of n samples from p discrete variables.
    ks : 1D array of length p, optional
        Alphabet size for each variable.
    r : int or None, optional; 1<=r<=p.
        If passed, return a generator yielding estimates for the p-choose-r
        possible combinations of r variables.
        If r == 1, return the entropy for each variable. If r == 2 return the
        mutual information for each possible pair. If r > 2 return the
        coinformation for each possible subset of length r.
        Combinations are ordered as: list(itertools.combinations(range(p), r)).

    Returns
    -------
    float
        Coinformation estimate.

    """

    # change sign for odd number of variables
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

    Raises
    ------
    CardinalityError
        If len(ks) != p.

    """

    from itertools import combinations

    # check data shape
    ar = _check_data(ar, axis)

    p = ar.shape[0]

    ks = _check_ks(ks, ar)
    if ks.ndim == 0:
        raise CardinalityError('ks cant be a scalar')

    if p > 2:
        h1 = list(from_data(ar, ks=ks, r=1))
        return (h1[i1] + h1[i2] - from_data(ar[[i1, i2]], ks=ks[[i1, i2]])
                for i1, i2 in combinations(range(p), 2))

    return numpy.sum(from_data(ar, ks=ks, r=1)) - from_data(ar, ks=ks)


def conditional_entropy(ar, c, ks=None, axis=1, r=None):
    """
    Coditional entropy estimate from data matrix.

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
    r : int or None, optional; 1<=r<=p-len(c).
        If passed, return a generator yielding estimates for all possible
        combinations of r variables conditioning on the `c` variables.
        Indices are sorted as:
        list(x for x in collections.combinations(range(p), r=r+len(c))
             if set(c) <= set(x))

    Returns
    -------
    float
        Conditional entropy estimate

    """
    from itertools import combinations

    # check data shape
    ar = _check_data(ar, axis)

    p = ar.shape[0]

    try:
        c = list(c)
    except TypeError:
        c = [c]
    if not set(c) <= set(range(p)):
        return EstimatorInputError('The indices of conditioning variables'
                                   ' are not valid')

    ks = _check_ks(ks, ar)

    # EntropyBasedEstimator objects are callable and return the fitted estimate
    estimator = Entropy()

    # Entropy of features on which we are conditioning
    counts = histogram(ar[c])
    hc = estimator(counts, k=ks)

    if r is not None:
        if ks.ndim == 0:
            raise CardinalityError('For combinations, ks cant be a scalar')

        r = _check_r(r, p - len(c))

        # include the c variables in the set
        r = r + len(c)

        indices = combinations(range(p), r=r)
        counts_combinations = histogram(ar, r=r)
        alphabet_size_combinations = (numpy.prod(x)
                                      for x in combinations(ks, r=r))
        return (estimator(pk, k=k) - hc for ids, pk, k in zip(
            indices, counts_combinations, alphabet_size_combinations)
                if set(c) <= set(ids))

    counts = histogram(ar)
    return estimator(counts, k=ks) - hc


def _nbins(data):
    """
    The number of unique elements along axis 0. If data is p-dimensional,
    the num. of unique elements for each variable.
    """
    # reshape as a p-by-n array
    return [len(numpy.unique(v)) for v in data]


def histogram(data, axis=1, r=None):
    """Compute an histogram from a data matrix. Wrapper to numpy.unique.

    Parameters
    ----------
    data : array-like, shape (p, n)
        A p-by-n array of n samples from p variables.
    axis : int, optional
        The sample-indexing axis. Defaults to 1.
    r : int, optional
        For r values in the interval [1, p],
        return a generator yielding bin counts for each of the p-choose-r
        combinations of r variables.

    Returns
    -------
    counts : ndarray
        Bin counts.

    """
    from itertools import combinations

    # check data shape
    data = _check_data(data, axis)

    if r is not None:
        r = _check_r(r, data)
        return (histogram(d) for d in combinations(data, r=r))

    # statistics for the p-dimensional variable
    _, counts = numpy.unique(data, return_counts=True, axis=1)
    return counts


def _check_data(ar, axis):
    """Check that input arrays are non-empty 2D arrays."""

    ar = numpy.atleast_2d(ar)
    if ar.ndim > 2:
        raise DataArrayError('Input array has %s dimensions; must be 2D' %
                             ar.ndim)
    p, n = ar.shape
    if n == 0 or p == 0:
        raise DataArrayError('Empty input array')

    if axis == 0:
        ar = ar.T

    return ar


def _check_r(r, ar):
    """
    Raises
    ------
    CombinationError
        For r values out of the interval [1, p].
    """
    if hasattr(ar, 'shape'):
        p = ar.shape[0]
    else:
        p = ar
    if r < 1 or r > p:
        raise CombinationError('r values must be in the interval [1, %s]' % p)
    return r


def _check_ks(ks, ar):
    """
    Raises
    ------
    CardinalityError
        If ks is array-like and len(ks) != p.
    """

    if ks is None:
        # guess from data
        ks = numpy.array([len(numpy.unique(v)) for v in ar])
    else:
        try:
            ks = numpy.float64(ks)
        except ValueError:
            raise CardinalityError('%s: not a valid cardinality')
        if ks.ndim:
            p = ar.shape[0]
            if len(ks) != p:
                raise CardinalityError('k should have len %s' % p)
    return ks


def iinfo(X, ks):
    """Helper function for interaction information definition.

    Ref: timme2014synergy
    """
    info = 0.0
    S = len(X)
    for T in range(1, S + 1):
        sgn = (-1)**(S - T)
        info += sgn * numpy.sum(from_data(X, ks=ks, r=T))
    return -info


def coinfo(X, ks):
    """Helper function for coinformation definition.

    Ref: timme2014synergy
    """
    info = 0.0
    S = len(X)
    for T in range(1, S + 1):
        sgn = (-1)**T
        info += sgn * numpy.sum(from_data(X, ks=ks, r=T))
    return -info


def is_pmf(a):
    """If a is a valid probability mass function."""
    a = numpy.float64(a)
    not_negative = numpy.all(a >= 0)
    normalized = numpy.isclose(sum(a), 1.0)
    return not_negative and normalized
