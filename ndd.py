# -*- coding: utf-8 -*-
# Copyright (C) 2016, Simone Marsili 
# All rights reserved.
# License: BSD 3 clause
"""
ndd
=====

Estimates of entropy and entropy-related quantities from discrete data.

Some refs:

Nemenman, I.; Shafee, F. Bialek, W. Entropy and inference, revisited. 
In Advances in Neural Information Processing Systems 14:471--478 (2002). 

Nemenman, I.; Bialek, W.; Van Steveninck, RDR. Entropy and information in neural spike trains: Progress on the sampling problem. 
Physical Review E, 69(5):056111 (2004).

Hutter, M. Distribution of mutual information. 
Advances in neural information processing systems, 1:399--406 (2002).

"""

from __future__ import absolute_import,division,print_function,unicode_literals
from builtins import *

__copyright__ = "Copyright (C) 2016 Simone Marsili"
__license__   = "BSD 3 clause"
__version__   = "v0.1.6"
__author__    = "Simone Marsili (simomarsili@gmail.com)"
__all__ = ['entropy','histogram']
import numpy as np
import warnings
import sys
import time

def _plugin(h):
    """
    Wrapper to the plugin estimator. 
    Compute an estimate P of the true probability distribution directly from the histogram counts, 
    and then compute the entropy from P. 
    
    Parameters
    ----------
    h   : list or array of ints
          histogram counts

    Returns
    -------
    val : float
          entropy estimate
    """
    import nddf
    h = np.array(h,dtype=np.int32)
    return nddf.plugin(h)

def _pseudo(h,k=None,alpha=0.0):
    """
    Wrapper to the pseudo-count estimator. 
    Compute an estimate P of the true probability distribution by adding alpha pseudocounts to the original counts, 
    and then compute the entropy from P. 
    
    Parameters
    ----------
    h     : list or array of ints
            histogram counts 
    k     : int (optional)
            number of categories (bins). If None, k will be guessed from the number of elements in histogram h
    alpha : float
            pseudocounts that will be added to the frequency of every possible category. Default value is 0 i.e. no pseudocounts. 

    Returns
    -------
    val   : float
            entropy estimate
    """
    import nddf
    h = np.array(h,dtype=np.int32)
    if k is None: k = len(h)
    k = np.int32(k)
    alpha = np.float64(alpha)
    return nddf.pseudo(h,k,alpha)

def _dirichlet(h,k=None,alpha=0.0):
    """
    Wrapper to the Dirichlet estimator. 
    The estimated entropy is the mean entropy of the ensemble of Dirichlet posterior distributions, 
    resulting from a (symmetric) Dirichlet prior having alpha as concentration parameter. 
    Notice that the probability distribution obtained adding alpha pseudo-counts to each category (pseudo-count estimator)
    corresponds to the average Dirichlet posterior distribution (using alpha as concentration parameter), 
    i.e. the maximum-likelihood estimate of the true distribution using a Dirichlet prior. 
    
    Parameters
    ----------
    h     : list or array of ints
            histogram counts 
    k     : int (optional)
            number of categories (bins). If None, k will be guessed from the number of elements in histogram h
    alpha : float
            pseudocounts that will be added to the frequency of every possible category. Default value is 0 i.e. no pseudocounts. 

    Returns
    -------
    val   : float
            entropy estimate
    """
    import nddf
    h = np.array(h,dtype=np.int32)
    if k is None: k = len(h)
    k = np.int32(k)
    alpha = np.float64(alpha)
    return nddf.dirichlet(h,k,alpha)

def _nsb(h,k=None):
    """
    Wrapper to the Nemenman-Shafee-Bialek (NSB) estimator. 
    The estimated entropy is an average over an infinite series of Dirichlet estimators. 
    The corresponding distribution of the concentration parameter alpha is chosen is such a way that the resulting prior is uninformative on the entropy. 
    From a practical point of view, the NSB estimator is a parameter-free, fully Bayesian estimator 
    that doesnt require the user to guess an (inevitably data-dependent) reasonable number of pseudo-counts or optimal value for the concentration parameter. 
    Moreover, Bayesian confidence intervals can be obtained from the variance of the entropy over the Dirichlet priors. 
    
    Parameters
    ----------
    h     : list or array of ints
            histogram counts 
    k     : int (optional)
            number of categories (bins). If None, k will be guessed from the number of elements in histogram h

    Returns
    -------
    val   : float
            entropy estimate
    """
    import nddf
    h = np.array(h,dtype=np.int32)
    if k is None: k = len(h)
    k = np.int32(k)
    estimate,std = nddf.nsb(h,k)
    return (estimate,std)

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
    
