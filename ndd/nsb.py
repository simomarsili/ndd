# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""
Functions for entropy and information measures estimation.
"""
import logging
from itertools import combinations

import numpy

from ndd.counter import Counts
from ndd.data import DataArray
from ndd.divergence import JSDivergence
from ndd.estimators import guess_alphabet_size  # pylint: disable=unused-import
from ndd.estimators import NSB, AutoEstimator, Plugin, check_estimator
from ndd.exceptions import EstimatorInputError, PmfError

# from ndd.failing import dump_on_fail

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


# @dump_on_fail()
# pylint: disable=line-too-long
def entropy(nk, k=None, zk=None, estimator=None, return_std=False):
    """
    Bayesian Entropy estimate from an array of counts.

    The `entropy` function takes as input a vector of frequency counts `nk`
    (the observed frequencies for a set of classes or states) and an alphabet
    size `k` (the number of classes with non-zero probability, including
    unobserved classes) and returns an entropy estimate (in nats) computed
    using the Nemenman-Schafee-Bialek (NSB) algorithm.

    >>> import ndd
    >>> counts = [4, 12, 4, 5, 3, 1, 5, 1, 2, 2, 2, 2, 11, 3, 4, 12, 12, 1, 2]
    >>> ndd.entropy(counts, k=100)
    2.8060922529931225

    The uncertainty in the entropy estimate can be quantified using the
    posterior standard deviation (see Eq. 13 in Archer 2013:
    https://pillowlab.princeton.edu/pubs/Archer13_MIestim_Entropy.pdf

    >>> ndd.entropy(counts, k=100, return_std=True)
    (2.8060922529931225, 0.11945501149743358)

    If the alphabet size is unknown or countably infinite, the `k` argument
    can be omitted and the function will either use an upper bound
    estimate for `k` or switch to the asymptotic NSB estimator for strongly
    undersampled distributions (see Equations 29, 30 in Nemenman2011:
    https://nemenmanlab.org/~ilya/images/c/c1/Nemenman_2011b.pdf)

    >>> import ndd
    >>> counts = [4, 12, 4, 5, 3, 1, 5, 1, 2, 2, 2, 2, 11, 3, 4, 12, 12, 1, 2]
    >>> ndd.entropy(counts)  # k is omitted
    2.8130746489179046

    When the alphabet size is unknown and no coincidences (no bins with
    counts > 1) can be found in the counts array, no entropy estimator can
    give a reasonable estimate. In this case, the function returns the
    logarithm of the number of samples, and the error is set to inf.

    >>> counts = [1]*100
    >>> ndd.entropy(counts, return_std=True)
    (4.605170185988092, inf)

    Parameters
    ----------
    nk : array-like or mapping
        The number of occurrences of a set of bins. For mappings
        use the dictionary values `nk.values()` as counts.
    k : int or array-like, optional
        Alphabet size (the number of bins with non-zero probability).
        If an array, set k = numpy.prod(k).
        Default: use an upper bound estimate for the alphabet size. If the
        distribution is strongly undersampled, switch to the asymptotic NSB
        estimator that can be used even if the alphabet size is unknown.
    zk : array_like, optional
        Counts distribution or "multiplicities". If passed, nk contains
        the observed counts values and len(zk) == len(nk).
    estimator : str or entropy estimator obj, optional
        Enforce a specific estimator. Check `ndd.entropy_estimators` for
        available estimators. Default: use the NSB estimator (or the zeroth
        order approximation discussed in Nemenman2011 if the distribution is
        strongly undersampled).
    return_std : boolean, optional
        If True, also return the standard deviation over the entropy posterior.

    Returns
    -------
    entropy : float
        Entropy estimate.
    err : float, optional
        Bayesian error bound on the entropy estimate.
        Only if `return_std` is True.

    Notes
    -----
    After a call, the fitted parameters can be inspected checking the
    `entropy.info` dictionary:

    >>> import ndd
    >>> counts = [4, 12, 4, 5, 3, 1, 5, 1, 2, 2, 2, 2, 11, 3, 4, 12, 12, 1, 2]
    >>> ndd.entropy(counts)
    2.8130746489179046
    >>> ndd.entropy.info
    {'entropy': 2.8130746489179046, 'err': 0.1244390183672502, 'bounded': 1, 'estimator': NSB(alpha=None), 'k': 6008}

    `entropy.info['bounded']` is equal to 1 if the entropy estimate has error
    bounds, 0 when the estimate is unbounded (when no coincidences have
    occurred in the data).

    """

    counts = Counts(nk=nk, zk=zk)
    nk, zk = counts.multiplicities

    if estimator is None:
        estimator = 'AutoEstimator'

    estimator, _ = check_estimator(estimator)

    estimator = estimator.fit(nk=nk, zk=zk, k=k)

    S, err = estimator.estimate_, estimator.err_

    if S is not None and numpy.isnan(S):
        logger.warning('nan value for entropy estimate')
        S = numpy.nan

    if err is not None and numpy.isnan(err):
        err = numpy.nan

    # annotate the entropy function
    entropy.info = {}
    entropy.info['entropy'] = S
    entropy.info['err'] = err
    entropy.info['bounded'] = int(err is not None and numpy.isfinite(err))
    if isinstance(estimator, AutoEstimator):
        entropy.info['estimator'] = str(estimator.estimator)
        entropy.info['k'] = estimator.k
    else:
        entropy.info['estimator'] = str(estimator)
        entropy.info['k'] = k

    if return_std:
        if err is not None and numpy.isnan(err):
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
        If int, the variables share the same alphabet size.
    estimator : str or entropy estimator instance, optional
        If a string, use the estimator class with the same name and default
        parameters. Check ndd.entropy_estimators for the available estimators.
        Default: Nemenman-Shafee-Bialek (NSB) estimator.
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
        return estimates_from_combinations(ar, r, estimator=estimator)

    counts, k = ar.iter_counts()
    return estimator(counts, k=k)


def jensen_shannon_divergence(nk, k=None, estimator='NSB'):
    """
    Jensen-Shannon divergence from a m-by-p matrix of counts.

    Return an estimate of the Jensen-Shannon divergence between
    m unknown discrete distributions from a m-by-p input array of counts.
    The estimate (in nats) is computed as a combination of single Bayesian
    entropy estimates. If the total number of samples varies among the
    distributions, use a weighted average (see the general definition of
    the Jensen-Shannon divergence:
    https://en.wikipedia.org/wiki/Jensen-Shannon_divergence).

    Parameters
    ----------

    nk : array-like, shape (m, p)
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

    estimator = JSDivergence(estimator).fit(nk, k=k)
    js = estimator.estimate_

    if numpy.isnan(js):
        logger.warning('nan value for JS divergence')
        js = numpy.nan

    return js


def cross_entropy(nk, qk):
    """
    Cross entropy: - sum(nk log(nk/qk))
    Parameters
    ----------
    nk : array_like
        Probability mass function. Normalize if doesnt sum to 1.
    qk : array_like
        Probability mass function. Must be len(qk) == len(nk).

    Returns
    -------
    float
        Cross entropy

    """

    if len(qk) != len(nk):
        raise PmfError('qk and nk must have the same length.')

    nk = numpy.asarray(nk)
    qk = numpy.asarray(qk)

    if any(nk < 0):
        raise PmfError('nk entries must be positive')
    if not is_pmf(qk):
        raise PmfError('qk must be a valid PMF (positive, normalized)')

    nk = 1.0 * nk / numpy.sum(nk)
    qk = numpy.log(1.0 * qk)

    return -numpy.sum(nk * qk)


def kullback_leibler_divergence(nk, qk, estimator='NSB'):
    """
    Kullback-Leibler divergence given counts nk and a reference PMF qk.

    Return an estimate of the Kullback-Leibler given an array of counts nk and
    a reference PMF qk. The estimate (in nats) is computed as:
    - S_p - sum(nk * log(qk)) / sum(nk)
    where S_p is the entropy estimate from counts nk.

    Parameters
    ----------
    nk : array_like
        The number of occurrences of a set of bins.
    qk : array_like
        Reference PMF in sum(nk log(nk/qk).
        Must be a valid PMF (non-negative, normalized) and len(qk) = len(nk).
    estimator : str or entropy estimator instance, optional
        If a string, use the estimator class with the same name and default
        parameters. Check ndd.entropy_estimators for the available estimators.
        Default: use the  Nemenman-Shafee-Bialek (NSB) estimator.

    Returns
    -------
    float
        Kullback-Leibler divergence.

    """

    estimator, _ = check_estimator(estimator)
    nk = numpy.asarray(nk)
    k = len(qk)
    if k == 1:  # single bin
        return 0.0

    return cross_entropy(nk, qk) - estimator.fit(nk, k=k).estimate_


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
        Default: Nemenman-Shafee-Bialek (NSB) estimator.
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
        return estimates_from_combinations(ar, r, q=iinfo, estimator=estimator)

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
        Default: Nemenman-Shafee-Bialek (NSB) estimator.
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

    change_sign = ar.shape[0] % 2
    iinfo = interaction_information(ar=ar,
                                    ks=ks,
                                    estimator=estimator,
                                    axis=axis,
                                    r=r)
    if change_sign:
        # change sign for odd number of variables
        if r is not None:
            result = (-ii for ii in iinfo)
        else:
            result = -iinfo  # pylint:disable=invalid-unary-operand-type
    else:
        result = iinfo

    return result


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

    # entropies for single fearures
    h1 = list(from_data(ar, r=1, estimator=estimator))

    if p > 2:

        def mi_for_all_pairs():
            """Yield the mutual info for all pairs of features."""
            for i1, i2 in combinations(range(p), 2):
                yield h1[i1] + h1[i2] - from_data(ar[i1, i2],
                                                  estimator=estimator)

        return mi_for_all_pairs()

    return h1[0] + h1[1] - from_data(ar, estimator=estimator)


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
    # TODO: add tests if r is not None

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
    counts, kc = ar[c].iter_counts()
    hc = estimator(counts, k=kc)

    def centropy(x, k, estimator):
        """Helper function for conditional entropy."""
        return from_data(x, ks=k, estimator=estimator) - hc

    if r is not None:

        # r should be >= p - len(c)

        # include the c variables in the set
        r = r + len(c)

        return estimates_from_combinations(ar,
                                           r,
                                           q=centropy,
                                           estimator=estimator,
                                           subset=c)

    _, k = ar.iter_data()
    return centropy(ar, k, estimator)


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
    normalized = numpy.isclose(numpy.sum(a), 1.0)
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


def estimates_from_combinations(ar,
                                r,
                                *,
                                q=from_data,
                                estimator='NSB',
                                subset=None):
    """Apply the estimator function `func` to combinations of data features.

    Parameters
    ----------
    ar : array-like
        p-by-n data array.
    r : int, optional; ; 1<=r<=p.
        Length of feature combinations.
    q : callable, optional
        A callable returning the quantity to be estimated from data
    estimator : str or entropy estimator instance, optional
        If a string, use the estimator class with the same name and default
        parameters. Check ndd.entropy_estimators for the available estimators.
        Default: Nemenman-Shafee-Bialek (NSB) estimator.
    subset : list, optional
        Return only the results for combinations of features that contain
        the `subset` of features.

    Yields
    ------
    estimate : float

    """
    estimator, _ = check_estimator(estimator)

    if not isinstance(ar, DataArray):
        ar = DataArray(ar)

    p = ar.shape[0]
    feature_combinations = zip(combinations(range(p), r), ar.iter_data(r=r))

    if subset is not None:
        subset = set(subset)
        for ids, (x, k) in feature_combinations:
            if subset <= set(ids):
                yield q(x, k, estimator=estimator)
    else:
        for ids, (x, k) in feature_combinations:
            yield q(x, k, estimator=estimator)
