# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""
ndd
===

Estimates of entropy and entropy-related quantities from discrete data.

Some refs:

Nemenman, I.; Shafee, F. Bialek, W.
Entropy and inference, revisited.
In Advances in Neural Information Processing Systems 14:471--478 (2002).

Nemenman, I.; Bialek, W.; Van Steveninck, RDR.
Entropy and information in neural spike trains: Progress on the sampling
problem.
Physical Review E, 69(5):056111 (2004).

Archer, Evan, Park, Il Memming and Pillow, Jonathan W.
Bayesian and Quasi-Bayesian Estimators for Mutual Information from Discrete
Data.
Entropy 15 , no. 5 (2013): 1738--1755.

Hutter, M.
Distribution of mutual information.
Advances in neural information processing systems, 1:399--406 (2002).

"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)
import time
import logging
from version import __version__

__copyright__ = "Copyright (C) 2016,2017 Simone Marsili"
__license__   = "BSD 3 clause"
__author__    = "Simone Marsili (simomarsili@gmail.com)"
__all__ = ['entropy', 'histogram']

import numpy as np
import warnings
import sys

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

    try:
        alpha = np.float64(alpha)
    except ValueError:
        raise

    return (counts, k, alpha)

def _plugin(counts):
    """Wrapper to the plugin estimator. Compute the max. likelihood estimate
    p_{ML} from the histogram counts, and then compute the entropy from p_{ML}
    as H(p_{ML}).

    Parameters
    ----------
    counts : array_like
        Histogram counts.

    Returns
    -------
    entropy : float
        Rntropy estimate.

    """
    import nddf

    counts, _, _ = _check_histogram(counts)
    return nddf.plugin(counts)

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

    k : int (optional)
        Total number of classes. must be k >= len(counts).
        Defaults to len(counts).

    alpha : float
        Sum alpha pseudocounts to the frequency of every bin.
        Default value is 0 (see 'plugin' estimator).

    Returns
    -------
    entropy : float
        entropy estimate

    """
    import nddf

    counts, k, alpha = _check_histogram(counts, k, alpha)
    return nddf.pseudo(counts, k, alpha)

def _dirichlet(counts, k=None, alpha=0.0):
    """ Wrapper to the Dirichlet estimator. Average over the posterior
    distribution for the entropy H(p), using a symmetric Dirichlet prior for
    the distribution p with concentration parameter `alpha`.

    Parameters
    ----------
    counts : array_like
        Histogram counts.

    k : int (optional)
        Total number of classes. must be k >= len(counts).
        Defaults to len(counts).

    alpha : float
        Concentration parameter of the Dirichlet prior.
        Must be >= 0.0. Defaults tp 0.0.

    Returns
    -------
    entropy : float
        entropy estimate

    """
    import nddf

    counts, k, alpha = _check_histogram(counts, k, alpha)
    return nddf.dirichlet(h,k,alpha)

def _nsb(counts, k=None):
    """Wrapper to the Nemenman-Shafee-Bialek (NSB) estimator.
    The entropy estimate results from an average over a distribution of
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

    k : int (optional)
        Total number of classes. must be k >= len(counts).
        Defaults to len(counts).

    Returns
    -------
    entropy : float
        entropy estimate

    """
    import nddf

    counts, k, _ = _check_histogram(counts, k)
    estimate, std = nddf.nsb(counts, k)
    return (estimate, std)

def entropy(h,algorithm='nsb',alpha=None,k=None,est_error=False,verbose=0):
    """
    Compute an estimate of the entropy for the histogram h
    
    Parameters
    ----------
    
    h         : list or array of ints
                histogram counts 
    algorithm : string, optional
                algorithm for entropy estimation. options are: 'nsb' (default), 'pseudo', 'dirichlet', 'plugin'. 
    alpha     : float, optional 
                pseudocounts to add to each of the histogram counts ('pseudo' estimator) or value of the concentration parameter ('dirichlet' estimator)
    k         : integer, optional 
                size of the alphabet or number of possible different outcomes (default is k = number of bins in histogram h)
    est_error : boolean, optional 
                if True, returns the tuple (ent,err) where err is a Bayesian standard error on the estimated entropy ent (default is False)
    verbose   : integer, optional 
                verbosity level
    
    Returns
    -------
    ent       : float
                estimated entropy 
    (err)     : float, optional
                estimated error on entropy ent (if est_error == True in input)

    """

    start_time = time.time()

    if not k: 
        k = len(h)
    else: 
        if len(h) > k: 
            raise ValueError('Improper input: k=%s is smaller than len(h)=%s' % (k,len(h)))

    if algorithm == 'pseudo' or algorithm == 'dirichlet': 
        if not alpha: 
            raise TypeError('Missing input: if algorithm = pseudo, dirichlet, alpha must be passed')
    if alpha: 
        if alpha <= 0.0: 
            raise ValueError('Improper input: alpha=%s must be >= 0' % alpha)
    if est_error and algorithm != 'nsb':
        raise ValueError('The error on the calculated entropy can be estimated only using the default algorithm ("nsb")')

    if algorithm == 'nsb'      : ent,err = _nsb(h,k)
    if algorithm == 'plugin'   : ent = _plugin(h)
    if algorithm == 'pseudo'   : ent = _pseudo(h,k,alpha)
    if algorithm == 'dirichlet': ent = _dirichlet(h,k,alpha)

    if verbose > 0: 
        print("(ndd.entropy) entropy estimation: %s secs" % (time.time() - start_time),file=sys.stderr)
        sys.stderr.flush()

    if est_error:
        return (ent,err)
    else:
        return ent

def histogram(data,dictionary=False,verbose=0):
    """
    Compute an histogram from data 
    
    Parameters
    ----------
    
    data      : hashable 
                data  
    verbose   : integer, optional 
                verbosity level
    
    Returns
    -------
    np.array  : np.int32
                histogram

    """
    import collections
    from collections import Counter    

    # rank of data must be either 1 or 2
    rank = len(np.asarray(data).shape)

    if rank == 1: 
        hist = Counter(data)
    elif rank == 2:
        hist = Counter([tuple(x) for x in data])
    else: 
        raise Exception('ndd.histog: rank of data (%s) cant be greater than 2' % rank)

    if verbose > 0: 
        print("(ndd.histogram) histogram data: %s secs" % (time.time() - start_time),file=sys.stderr)
        sys.stderr.flush()

    if dictionary:
        return hist
    else: 
        return list(hist.values())
    
