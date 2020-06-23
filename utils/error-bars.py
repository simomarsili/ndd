# -*- coding: utf-8 -*-
"""Plot estimates with error bars for random datasets."""
import matplotlib.pyplot as plt
import numpy
from numpy.random import dirichlet, multinomial
from scipy.stats import entropy

import ndd
import ndd.estimators

N = 1000
K = 1000000
ALPHA = 0.001
NITER = 1
R = 100

numpy.random.seed(1234)


def data(a=ALPHA):
    """
    Generate a random distribution p ~ Dirichlet([a]*K) and a random set of
    counts ~ Multinomial(N, p).
    """
    pp = dirichlet([a] * K)
    counts = multinomial(N, pp)
    counts = counts[numpy.nonzero(counts)]
    return counts, pp


def estimate(counts, pp, k=None, estimator='NSB'):
    """
    Return:
    * the true entropy of p
    * the entropy estimate
    * the estimated error
    """
    # reference value, estimate and error estimate
    ref = entropy(pp)
    result = ref, *ndd.entropy(
        counts, k=k, return_std=True, estimator=estimator)
    result = numpy.asarray(result)
    result = [x / numpy.log(K) if x is not None else 0.1 for x in result]
    return result


def plot_errorbar(ax, a):
    """Plotter"""
    kwargs = {
        'marker': 'o',
        'yerr': a[2],
        'alpha': 0.5,
        'elinewidth': 1,
        'capsize': 3,
        'errorevery': 1,
        'ls': '',
    }
    if len(a) == 2:
        kwargs.pop('yerr')
    ax.errorbar(a[0], a[1], **kwargs, zorder=1)
    ax.plot(a[0], a[0], '-', lw=2, zorder=2)
    return ax


if __name__ == '__main__':
    from time import time
    X = [data(a) for a in numpy.logspace(-3, 0, R) for _ in range(NITER)]

    fig, axs = plt.subplots(1, 2)

    t0 = time()
    ar0 = [estimate(*x, k=K) for x in X]
    print('time0: ', time() - t0)
    ar0 = numpy.array(list(zip(*ar0)))
    plot_errorbar(axs[0], ar0)

    t0 = time()
    ar1 = [estimate(*x, estimator='auto') for x in X]
    print('time0: ', time() - t0)
    ar1 = numpy.array(list(zip(*ar1)))
    plot_errorbar(axs[1], ar1)

    plt.show()

    # compute bias and std for k-guess
    ar0[1] = ar0[1] - ar0[0]
    ar1[1] = ar1[1] - ar1[0]
    print(numpy.mean(ar0[1]), numpy.std(ar0[1]))
    print(numpy.mean(ar1[1]), numpy.std(ar1[1]))
