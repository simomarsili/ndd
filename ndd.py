# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""
entropy from discrete data.

The **ndd** module is a simple and minimal Python interface to the
Nemenman-Schafee-Bialek (NSB) algorithm, a parameter-free, fully Bayesian
algorithm for entropy estimation from discrete data.

### Basic usage 

The `ndd.entropy` function takes as input an histogram vecor of counts
(a list/array-like of integers) and returns a entropy estimate computed as a
posterior mean (in nats): 

```python
>>> counts
[7, 3, 5, 8, 9, 1, 3, 3, 1, 0, 2, 5, 2, 11, 4, 23, 5, 0, 8, 0]
>>> import ndd
>>> entropy_estimate = ndd.entropy(counts)
>>> entropy_estimate
2.623634344902917
```

The uncertainty in the entropy estimate can be quantified by the posterior
standard deviation:
```python
>>> entropy_estimate, std = ndd.entropy(counts, return_std=True)
>>> std
0.048675500725595504
```

### References

Some refs:

```
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
```

and interesting links:

- [Sebastian Nowozin on Bayesian estimators](http://www.nowozin.net/sebastian/blog/estimating-discrete-entropy-part-3.html)

- [Il Memming Park on discrete entropy estimators](https://memming.wordpress.com/2014/02/09/a-guide-to-discrete-entropy-estimators/)

"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)

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

def entropy(counts, k=None, a=None, return_std=False, dist=False):
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
    pseudocounts to the empirical bin frequencies (`pseudocount` estimator).

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

    return_std : boolean, optional
        If True, also return the standard deviation over the posterior for H.

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

    std : float, optional
        If return_std == True, return the standard deviation over the posterior
        for H. When dist == True, return None.

    """
    import nddf

    counts, k, alpha = _check_histogram(counts, k, a)
    std = None
    if dist:
        if alpha < 1e-6:
            # we'll take this as zero
            estimate = nddf.plugin(counts)
        else:
            estimate = nddf.pseudo(counts, k, alpha)
    else:
        if a is None:
            # NSB
            estimate, std = nddf.nsb(counts, k)
        else:
            # fixed alpha
            estimate = nddf.dirichlet(counts, k, alpha)
            #TODO: compute variance over the posterior at fixed alpha

    if estimate is np.nan:
        raise FloatingPointError("Estimate is NaN")

    if return_std:
        return (estimate, std)
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
