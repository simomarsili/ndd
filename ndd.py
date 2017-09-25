# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""
ndd
===

Estimates of entropy and entropy-related quantities from discrete data.

Some basic refs:

@article{wolpert1995estimating,
  title={Estimating functions of probability distributions from a finite set of samples},
  author={Wolpert, David H and Wolf, David R},
  journal={Physical Review E},
  volume={52},
  number={6},
  pages={6841},
  year={1995},
  publisher={APS}
}

@inproceedings{nemenman2002entropy,
  title={Entropy and inference, revisited},
  author={Nemenman, Ilya and Shafee, Fariel and Bialek, William},
  booktitle={Advances in neural information processing systems},
  pages={471--478},
  year={2002}
}

@article{nemenman2004entropy,
  title={Entropy and information in neural spike trains: Progress on the sampling problem},
  author={Nemenman, Ilya and Bialek, William and van Steveninck, Rob de Ruyter},
  journal={Physical Review E},
  volume={69},
  number={5},
  pages={056111},
  year={2004},
  publisher={APS}
}

@article{archer2013bayesian,
  title={Bayesian and quasi-Bayesian estimators for mutual information from discrete data},
  author={Archer, Evan and Park, Il Memming and Pillow, Jonathan W},
  journal={Entropy},
  volume={15},
  number={5},
  pages={1738--1755},
  year={2013},
  publisher={Multidisciplinary Digital Publishing Institute}
}

"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)
from version import __version__

__copyright__ = "Copyright (C) 2016,2017 Simone Marsili"
__license__ = "BSD 3 clause"
__author__ = "Simone Marsili (simomarsili@gmail.com)"
__all__ = ['entropy', 'histogram']

import numpy as np

def _check_histogram(counts, k=None, alpha=0.0):
    """Check that `counts` contains valid frequency counts."""

    try:
        counts = np.array(counts, dtype=np.int32)
    except ValueError:
        raise
    if np.any(counts < 0):
        raise ValueError("A bin cant have a frequency < 0")

    nbins = np.int32(len(counts))
    if k is None:
        k = nbins
    else:
        try:
            k = np.int32(k)
        except ValueError:
            raise
        if k < nbins:
            raise ValueError("k (%s) is smaller than the number of bins (%s)"
                             % (k, nbins))
    if alpha is None:
        alpha = 0.0
    else:
        try:
            alpha = np.float64(alpha)
        except ValueError:
            raise

    return (counts, k, alpha)

def _pseudo(counts, k=None, alpha=0.0):
    """Wrapper to the pseudo-count estimator. Compute an estimate <p> of the
    true probability distribution by adding `alpha` pseudocounts to each bin
    and compute the entropy from <p>, H(<p>). The procedure has a Bayesian
    interpretation: <p> corresponds to the average distribution over the
    posterior resulting from a symmetric Dirichlet prior with concentration
    parameter `alpha`.

    Parameters
    ----------
    counts : array_like
        Histogram counts.

    k : int, optional
        Total number of classes. must be k >= len(counts).
        Defaults to len(counts).

    alpha : float, optional
        Sum alpha pseudocounts to the frequency of every bin.
        Defaults to 0.0 (no pseudocounts, ML estimator).

    Returns
    -------
    entropy : float
        Entropy estimate.

    """
    import nddf

    counts, k, alpha = _check_histogram(counts, k, alpha)
    if alpha < 1e-6:
        # well take this as zero
        return nddf.plugin(counts)
    else:
        return nddf.pseudo(counts, k, alpha)

def _dirichlet(counts, k=None, alpha=None):
    """ If alpha is not None:
    average over the posterior distribution for the entropy H(p),
    using a symmetric Dirichlet prior for
    the distribution p with concentration parameter `alpha`.

    If alpha is None, use the Nemenman-Shafee-Bialek (NSB) estimator.
    The entropy is estimated as an average over a distribution of
    Dirichlet estimators, parametrized by the concentration parameter alpha.
    The prior for alpha is chosen such that the resulting prior for the entropy
    is flat. In practice, the NSB estimator is a parameter-free, fully Bayesian
    estimator that doesnt require the user to guess
    (unavoidably case-dependent) values for pseudocounts or concetration
    parameter. Bayesian confidence intervals can be obtained from the variance
    of the entropy over the distribution of Dirichlet priors.

    Parameters
    ----------
    counts : array_like
        Histogram counts.

    k : int, optional
        Total number of classes. must be k >= len(counts).
        Defaults to len(counts).

    alpha : float, optional
        Concentration parameter of the Dirichlet prior.
        Must be >= 0.0. If None, defaults to the NSB estimator.

    Returns
    -------
    entropy : float
        Entropy estimate.

    error : float, optional
        Bayesian confidence interval as entropy +- error

    """
    import nddf

    if alpha is None:
        # NSB
        counts, k, alpha = _check_histogram(counts, k, alpha)
        estimate, error = nddf.nsb(counts, k)
    else:
        # fixed alpha
        counts, k, alpha = _check_histogram(counts, k, alpha)
        estimate = nddf.dirichlet(counts, k, alpha)
        error = 0.0 #TODO: compute variance over the posterior at fixed alpha
    return (estimate, error)

def entropy(counts, k=None, a=None, return_error=False, dist=False):
    """
    Compute an estimate of the entropy for the histogram h

    Parameters
    ----------

    counts : array_like
        Histogram counts.

    k : int, optional
        Total number of classes. must be k >= len(counts).
        Defaults to len(counts).

    a : float, optional
        Concentration parameter of the Dirichlet prior.
        Must be >= 0.0. If no value is passed, use a mixture of Dirichlet
        prior (Nemenman-Schafee-Bialek algorithm).

    return_error : boolean, optional
        If True, also return the Bayesian confidence intervals for the entropy
        estimate, as the std deviation over the posterior for H.

    dist : boolean, optional
        If True, the true underlying distribution is estimated from counts,
        and used in the entropy definition ("plugin" estimator).
        Use `a` is as concentration parameter for the Dirichlet prior.
        If `a` is None, use the empirical distribution (ML estimate).

    Returns
    -------
    entropy : float
        Entropy estimate.

    error : float, optional
        If return_error == True, return a Bayesian confidence interval.
        When dist == True, return None.

    """
    import nddf

    counts, k, alpha = _check_histogram(counts, k, a)
    error = None
    if dist:
        if alpha < 1e-6:
            # we'll take this as zero
            return nddf.plugin(counts)
        else:
            return nddf.pseudo(counts, k, alpha)
    else:
        if a is None:
            # NSB
            estimate, error = nddf.nsb(counts, k)
        else:
            # fixed alpha
            estimate = nddf.dirichlet(counts, k, alpha)
            #TODO: compute variance over the posterior at fixed alpha

    if return_error:
        return (estimate, error)
    else:
        return estimate

def histogram(data, return_unique=False):
    """Compute an histogram from data. Wrapper to numpy.unique.

    Parameters
    ----------

    data : array_like
        Input data array. If n-dimensional, statistics is computed along
        axis 0.

    return_unique : bool, optional
        If True, also return the unique elements corresponding to each bin.

    Returns
    -------

    counts : ndarray
        Bin counts.

    unique : ndarray, optional
        Unique elements corresponding to each bin in counts.
        Only if return_elements == True

    """

    unique, counts = np.unique(data, axis=0, return_counts=True)

    if return_unique:
        return (unique, counts)
    else:
        return counts
