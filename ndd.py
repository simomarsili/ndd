# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""
ndd - estimates of eNtropy from Discrete Data.
===

This module is a Python interface to the Nemenman-Schafee-Bialek (NSB) entropy
estimator[nemenman2002entropy, nemenman2004entropy], a parameter-free, fully
Bayesian algorithm. Entropy is estimated by averaging over a mixture of
Dirichlet estimators[wolpert1995estimating] with an uninformative hyper-prior
for the concentration parameter.

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

and interesting links on the web:

Sebastian Nowozin on Bayesian estimators:
http://www.nowozin.net/sebastian/blog/estimating-discrete-entropy-part-3.html

Il Memming Park on discrete entropy estimators:
https://memming.wordpress.com/2014/02/09/a-guide-to-discrete-entropy-estimators/

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

def entropy(counts, k=None, a=None, return_error=False, dist=False):
    """
    Compute an estimate of the entropy from histogram counts.
    If `a` is passed, compute a Bayesian estimate of the entropy using a single
    Dirichlet prior with concentration parameter `a` (fixed alpha estimator).
    If `a` is None, average over a mixture of Dirichlet estimators weighted by
    an uninformative hyper-prior (NSB estimator).
    Finally, if `dist` == True, first estimate the underlying distribution over
    states/classes and then plug this estimate into the entropy definition
    (maximum likelihood estimator). If `a` is passed in combination with
    `dist=True`, the true distribution is approximated by adding `a`
    pseudocunts to the empirical bin frequencies (`pseudocount` estimator).

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
        priors (Nemenman-Schafee-Bialek algorithm).

    return_error : boolean, optional
        If True, also return the Bayesian confidence intervals for the entropy
        estimate, as the std deviation over the posterior for H.

    dist : boolean, optional
        If True, the true underlying distribution is estimated from counts,
        and plugged in the entropy definition ("plugin" estimator).
        Use `a` as the concentration parameter for the Dirichlet prior
        ("pseudocount" estimator).
        If `a` is None, use the empirical distribution ("ML" estimatator).

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
