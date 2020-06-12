# -*- coding: utf-8 -*-
"""Check error estimation"""
import numpy as np
from numpy.random import dirichlet, multinomial
from scipy.stats import entropy as sp_entropy

import ndd

R = 100  # number of repetitions
N = 100000
K = 1000000
ALPHA = 1.0

np.random.seed(42)


def error(alpha, n):
    """Return the actual error and the estimated uncertainty (normalized)"""
    k = len(alpha)
    pvals = dirichlet(alpha)
    counts = multinomial(n, pvals)
    h0 = sp_entropy(pvals)
    h, std = ndd.entropy(counts, k=k, return_std=True)
    return (h - h0) / h0, std / h0


if __name__ == '__main__':
    for N in np.logspace(2, 6, 100):
        print(*error([ALPHA] * K, N))
