# -*- coding: utf-8 -*-
"""Plot estimates with error bars for random datasets."""
import matplotlib.pyplot as plt
import numpy
from numpy.random import dirichlet, multinomial
from scipy.stats import entropy

import ndd

N = 1000
K = 1000000
ALPHA = 0.001
NITER = 1

numpy.random.seed(1234)


def estimate(a=ALPHA):
    """
    Generate a random distribution p ~ Dirichlet([a]*K) and a random set of
    counts ~ Multinomial(N, p).
    Return:
    * the true entropy of p
    * the entropy estimate
    * the estimated error
    """
    pp = dirichlet([a] * K)
    ref = entropy(pp)
    counts = multinomial(N, pp)
    # reference value, estimate and error estimate
    result = ref, *ndd.entropy(counts, k=K, return_std=True)
    result = numpy.asarray(result)
    return result / numpy.log(K)


if __name__ == '__main__':
    data = [
        estimate(a) for a in numpy.logspace(-3, 0, 400) for _ in range(NITER)
    ]
    data = numpy.array(list(zip(*data)))
    # data[0] = numpy.abs(data[0] - data[1])
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(data[0], data[0], '-')
    axs[0].plot(data[0], data[1], 'o', alpha=0.2)
    axs[1].plot(data[0], data[0], '-')
    axs[1].errorbar(data[0],
                    data[1],
                    marker='o',
                    yerr=data[2],
                    alpha=0.5,
                    elinewidth=1,
                    capsize=3,
                    errorevery=10,
                    ls='')
    plt.show()
