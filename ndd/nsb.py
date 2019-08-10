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
from ndd.estimators import NSB, Plugin, check_estimator
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


def entropy(pk, k=None, estimator='NSB', return_std=False):
    """
    Entropy estimate from an array of counts.

    Return a Bayesian estimate for the entropy of an unknown discrete
    distribution from an input array of counts pk.

    Parameters
    ----------
    pk : array-like
        The number of occurrences of a set of bins.
    k : int or array-like, optional
        Alphabet size (the number of bins with non-zero probability).
        Must be >= len(pk). A float is a valid input for whole numbers
        (e.g. k=1.e3). If an array, set k = numpy.prod(k).
        Default: k = sum(pk > 0)
    estimator : str or entropy estimator instance, optional
        If a string, use the estimator class with the same name and default
        parameters. Check ndd.entropy_estimators for the available estimators.
        Default: use the  Nemenman-Shafee-Bialek (NSB) estimator.
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

    estimator, _ = check_estimator(estimator)

    pk = numpy.asarray(pk)
    if k is None:
        k = sum(pk > 0)

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


def from_data(ar, ks=None, estimator='NSB', axis=0, r=None):
    """
    Entropy estimate from data matrix.

    Paramaters
    ----------
    ar : array-like, shape (n, p)
        2D array of n samples from p discrete variables.
    ks : int or 1D array of length p, optional
        Alphabet size for each variable.
    estimator : str or entropy estimator instance, optional
        If a string, use the estimator class with the same name and default
        parameters. Check ndd.entropy_estimators for the available estimators.
        Default: use the  Nemenman-Shafee-Bialek (NSB) estimator.
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

    estimator, _ = check_estimator(estimator)

    if not isinstance(ar, DataArray):
        ar = DataArray(ar, ks=ks, axis=axis)

    if r is not None:
        return (estimator(pk, k=k) for pk, k in ar.iter_counts(r=r))

    counts, k = ar.iter_counts()
    return estimator(counts, k=k)


def jensen_shannon_divergence(pk, k=None, estimator='NSB'):
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
    estimator : str or entropy estimator instance, optional
        If a string, use the estimator class with the same name and default
        parameters. Check ndd.entropy_estimators for the available estimators.
        Default: use the  Nemenman-Shafee-Bialek (NSB) estimator.

    Returns
    -------
    float
        Jensen-Shannon divergence.

    """

    estimator, _ = check_estimator(estimator)

    estimator = JSDivergence(estimator).fit(pk, k=k)
    js = estimator.estimate_

    if numpy.isnan(js):
        logger.warning('nan value for JS divergence')
        js = numpy.nan

    return js


def kullback_leibler_divergence(pk, qk, k=None, estimator='NSB'):
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
    estimator : str or entropy estimator instance, optional
        If a string, use the estimator class with the same name and default
        parameters. Check ndd.entropy_estimators for the available estimators.
        Default: use the  Nemenman-Shafee-Bialek (NSB) estimator.

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

    estimator, _ = check_estimator(estimator)
    estimate = estimator.fit(pk, k=k).estimate_
    kl = -(estimate + numpy.sum(pk * log_qk) / float(sum(pk)))
    if numpy.isnan(kl):
        logger.warning('nan value for KL divergence')
        kl = numpy.nan

    return kl


def interaction_information(ar, ks=None, estimator='NSB', axis=0, r=None):
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
    estimator : str or entropy estimator instance, optional
        If a string, use the estimator class with the same name and default
        parameters. Check ndd.entropy_estimators for the available estimators.
        Default: use the  Nemenman-Shafee-Bialek (NSB) estimator.
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

    def iinfo(X, ks, estimator):
        """Helper function for interaction information definition.

        Ref: timme2014synergy
        """
        info = 0.0
        S = len(X)
        for T in range(1, S + 1):
            sgn = (-1)**(S - T)
            info += sgn * sum(from_data(X, ks=ks, estimator=estimator, r=T))
        return -info

    estimator, _ = check_estimator(estimator)

    if not isinstance(ar, DataArray):
        ar = DataArray(ar, ks=ks, axis=axis)

    if r is not None:
        return (iinfo(data, k, estimator) for data, k in ar.iter_data(r=r))

    data, k = ar.iter_data()
    return iinfo(data, k, estimator)


def coinformation(ar, ks=None, estimator='NSB', axis=0, r=None):
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
    estimator : str or entropy estimator instance, optional
        If a string, use the estimator class with the same name and default
        parameters. Check ndd.entropy_estimators for the available estimators.
        Default: use the  Nemenman-Shafee-Bialek (NSB) estimator.
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
        ar=ar, ks=ks, estimator=estimator, axis=axis, r=r)


def mutual_information(ar, ks=None, estimator='NSB', axis=0):
    """Mutual information from p-by-n data matrix.

    If p > 2, return an estimate of the mutual information for each possible
    pair of variables, ordered as list(itertools.combinations(range(p), r=2)).

    Paramaters
    ----------
    ar : array-like
        n-by-p array of n samples from p discrete variables.
    ks : 1D p-dimensional array, optional
        Alphabet size for each variable.
    estimator : str or entropy estimator instance, optional
        If a string, use the estimator class with the same name and default
        parameters. Check ndd.entropy_estimators for the available estimators.
        Default: use the  Nemenman-Shafee-Bialek (NSB) estimator.
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
    estimator, _ = check_estimator(estimator)

    if not isinstance(ar, DataArray):
        ar = DataArray(ar, ks=ks, axis=axis)

    p = ar.shape[0]

    if p > 2:
        h1 = list(from_data(ar, r=1))
        return (h1[i1] + h1[i2] - from_data(ar[i1, i2], estimator=estimator)
                for i1, i2 in combinations(range(p), 2))

    return (sum(from_data(ar, r=1, estimator=estimator)) -
            from_data(ar, estimator=estimator))


def conditional_entropy(ar, c, ks=None, estimator='NSB', axis=0, r=None):  # pylint: disable=too-many-arguments
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
    estimator : str or entropy estimator instance, optional
        If a string, use the estimator class with the same name and default
        parameters. Check ndd.entropy_estimators for the available estimators.
        Default: use the  Nemenman-Shafee-Bialek (NSB) estimator.
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
        ar = DataArray(ar, ks=ks, axis=axis)

    p = ar.shape[0]

    try:
        c = list(c)
    except TypeError:
        c = [c]
    if not set(c) <= set(range(p)):
        return EstimatorInputError('The indices of conditioning variables'
                                   ' are not valid')

    # EntropyEstimator objects are callable and return the fitted estimate
    estimator, _ = check_estimator(estimator)

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
            estimator = NSB(alpha)
    return estimator
