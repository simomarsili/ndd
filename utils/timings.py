# -*- coding: utf-8 -*-
"""Measure execution average execution times"""
import cProfile
import time

import numpy as np
from numpy.random import dirichlet, multinomial
from scipy.stats import entropy as sp_entropy

import ndd
import ndd.fnsb  # pylint: disable=no-name-in-module, import-error

R = 200  # number of repetitions
N = 10000
K = 1000000
ALPHA = 0.1

np.random.seed(42)


def dimul(alpha, n, size=None):
    """Dirichlet-multinomial"""
    pvals = dirichlet(alpha)
    return multinomial(n, pvals, size=size)


def entropy(counts, k):
    """ndd.entropy() execution time"""
    start = time.time()
    _ = ndd.entropy(counts, k=k, return_std=True)
    end = time.time()
    return end - start, 0


def scipy_entropy(counts, k):  # pylint: disable=unused-argument
    """scipy.stats.entropy() execution time"""
    start = time.time()
    _ = sp_entropy(counts)
    end = time.time()
    return end - start, 0


def average_timings(ar):
    """Average execution times."""
    # pylint: disable=no-member
    labels = 'init range fortran python scipy_entropy'.split()
    funcs = (ndd.fnsb.phony_1, ndd.fnsb.phony_2, ndd.fnsb.phony_4, entropy,
             scipy_entropy)
    times = np.zeros(len(funcs))
    for counts in ar:
        for j, f in enumerate(funcs):
            times[j] += f(counts, K)[0]
    times /= R

    print('total fortran/python/scipy time: %e/%e/%e' % tuple(times[-3:]))
    for j, label in enumerate(labels):
        t0 = times[j - 1] if j > 0 else 0
        print('%s: %e' % (label, times[j] - t0))


def cprofile(ar):
    """Run cprofile"""
    return [ndd.entropy(x, k=K) for x in ar]


if __name__ == '__main__':
    # run cProfile
    a = dimul([ALPHA] * K, N, size=(R))
    cProfile.run('cprofile(a)')
    average_timings(a)
