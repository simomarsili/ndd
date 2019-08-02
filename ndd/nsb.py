# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""
Functions for entropy and information measures estimation.
"""
import logging
from itertools import combinations

import numpy

from ndd.data import DataArray
from ndd.divergence import JSDivergence
from ndd.estimators import NSB, Plugin, WolpertWolf
from ndd.exceptions import EstimatorInputError, PmfError

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

    estimator = select_estimator(alpha=alpha, plugin=plugin)

    if k is None:
        algorithm = type(estimator).__name__
        if algorithm in {'NSB', 'WolpertWolf'}:
            logger.warning(
                'WARNING: k=None but %s est. needs alphabet size; '
                'set k=len(pk)', algorithm)
        k = len(pk)

    estimator = estimator.fit(pk, k=k)
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


def from_data(ar, ks=None, axis=0, r=None):
    """
    Entropy estimate from data matrix.

    Paramaters
    ----------
    ar : array-like, shape (n, p)
        2D array of n samples from p discrete variables.
    ks : int or 1D array of length p, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis. Defaults to 0.
    r : int, optional; ; 1<=r<=p.
        If passed, return a generator yielding estimates for the p-choose-r
        possible combinations of r variables.

    Returns
    -------
    float
        Entropy estimate

    """
    if not isinstance(ar, DataArray):
        ar = DataArray(ar, k=ks, axis=axis)

    # EntropyEstimator objects are callable and return the fitted estimate
    estimator = NSB()

    if r is not None:
        return (estimator(pk, k=k) for pk, k in ar.iter_counts(r=r))

    counts, k = ar.iter_counts()
    return estimator(counts, k=k)


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

    entropy_estimator = select_estimator(alpha, plugin)
    estimator = JSDivergence(entropy_estimator).fit(pk, k=k)
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

    if sum(numpy.isinf(log_qk)) > 0:
        raise PmfError('qk must be positive')

    if len(log_qk) != len(pk):
        raise PmfError('qk and pk must have the same length.')

    if k == 1:  # single bin
        return 0.0
    if k is None:
        k = len(pk)

    estimator = select_estimator(alpha, plugin)
    estimate = estimator.fit(pk, k=k).estimate_
    kl = -(estimate + numpy.sum(pk * log_qk) / float(sum(pk)))
    if numpy.isnan(kl):
        logger.warning('nan value for KL divergence')
        kl = numpy.nan

    return kl


def interaction_information(ar, ks=None, axis=0, r=None):
    """Interaction information from data matrix.

    See Eq.10 in:
    Timme, Nicholas, et al.
    "Synergy, redundancy, and multivariate information measures:
    an experimentalist's perspective."
    Journal of computational neuroscience 36.2 (2014): 119-140.

    Paramaters
    ----------
    ar : array-like
        n-by-p array of n samples from p discrete variables.
    ks : 1D array of length p, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis. Defaults to 0.
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
    if not isinstance(ar, DataArray):
        ar = DataArray(ar, k=ks, axis=axis)

    if r is not None:
        return (iinfo(data, k) for data, k in ar.iter_data(r=r))

    data, k = ar.iter_data()
    return iinfo(data, k)


def coinformation(ar, ks=None, axis=0, r=None):
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
        n-by-p array of n samples from p discrete variables.
    ks : 1D array of length p, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis. Defaults to 0.
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
    return (-1)**ar.shape[0] * interaction_information(
        ar=ar, ks=ks, axis=axis, r=r)


def mutual_information(ar, ks=None, axis=0):
    """Mutual information from p-by-n data matrix.

    If p > 2, return an estimate of the mutual information for each possible
    pair of variables, ordered as list(itertools.combinations(range(p), r=2)).

    Paramaters
    ----------
    ar : array-like
        n-by-p array of n samples from p discrete variables.
    ks : 1D p-dimensional array, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis. Defaults to 0.

    Returns
    -------
    float
        Coinformation estimate.

    Raises
    ------
    CardinalityError
        If len(ks) != p.

    """
    if not isinstance(ar, DataArray):
        ar = DataArray(ar, k=ks, axis=axis)

    p = ar.shape[0]

    if p > 2:
        h1 = list(from_data(ar, r=1))
        return (h1[i1] + h1[i2] - from_data(ar[i1, i2])
                for i1, i2 in combinations(range(p), 2))

    return numpy.sum(from_data(ar, r=1)) - from_data(ar)


def conditional_entropy(ar, c, ks=None, axis=0, r=None):
    """
    Coditional entropy estimate from data matrix.

    Paramaters
    ----------
    ar : array-like
        n-by-p array of n samples from p discrete variables.
    c : int or array-like
        The variables on which entropy is conditioned (as column indices).
    ks : 1D p-dimensional array, optional
        Alphabet size for each variable.
    axis : int, optional
        The sample-indexing axis. Defaults to 0.
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
    # check data shape
    if not isinstance(ar, DataArray):
        ar = DataArray(ar, k=ks, axis=axis)

    p = ar.shape[0]

    try:
        c = list(c)
    except TypeError:
        c = [c]
    if not set(c) <= set(range(p)):
        return EstimatorInputError('The indices of conditioning variables'
                                   ' are not valid')

    # EntropyEstimator objects are callable and return the fitted estimate
    estimator = NSB()

    # Entropy of features on which we are conditioning
    counts, k = ar[c].iter_counts()
    hc = estimator(counts, k=k)

    if r is not None:

        # r should be >= p - len(c)

        # include the c variables in the set
        r = r + len(c)

        indices = combinations(range(p), r=r)

        return (estimator(counts, k=k) - hc
                for ids, (counts, k) in zip(indices, ar.iter_counts(r=r))
                if set(c) <= set(ids))

    counts, k = ar.iter_counts()
    return estimator(counts, k=k) - hc


def histogram(data, axis=0, r=None):
    """Compute an histogram from a data matrix. Wrapper to numpy.unique.

    Parameters
    ----------
    data : array-like, shape (p, n)
        A n-by-p array of n samples from p variables.
    axis : int, optional
        The sample-indexing axis. Defaults to 0.
    r : int, optional
        For r values in the interval [1, p],
        return a generator yielding bin counts for each of the p-choose-r
        combinations of r variables.

    Returns
    -------
    counts : ndarray
        Bin counts.

    """
    if not isinstance(data, DataArray):
        data = DataArray(data, axis=axis)

    if r is not None:
        return (histogram(d) for d in combinations(data, r=r))

    # statistics for the p-dimensional variable
    _, counts = numpy.unique(data, return_counts=True, axis=1)
    return counts


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


def select_estimator(alpha, plugin):
    """Select the appropriate estimator."""
    if plugin:
        estimator = Plugin(alpha)
    else:
        if alpha is None:
            estimator = NSB()
        else:
            estimator = WolpertWolf(alpha)
    return estimator
