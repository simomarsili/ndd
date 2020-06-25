# -*- coding: utf-8 -*-
"""Plot estimates with error bars for random datasets."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
from numpy.random import dirichlet, multinomial
from scipy.stats import entropy

import figs
import ndd
import ndd.estimators
from ndd.exceptions import NddError

N = 100
K = 10000
ALPHA = 0.001
NITER = 10
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


def estimate(counts, pp, k=None, estimator=None):
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
    if None in result:
        return None
    result = [x / numpy.log(K) for x in result]
    return result


def plot_errorbar(axes, a, label=None):
    """Plotter"""
    kwargs = {
        #'markersize': 4,
        'marker': 'o',
        'alpha': 0.5,
        'elinewidth': 1,
        'capsize': 3,
        'errorevery': 1,
        'ls': ''
    }
    if len(a) == 3:
        kwargs['yerr'] = a[2]
    axes.errorbar(a[0], a[1], **kwargs, zorder=1, label=label)
    axes.plot(a[0], a[0], '--', lw=1, zorder=2, color='black', alpha=0.5)
    return axes


if __name__ == '__main__':
    from time import time
    ylim = (-0.1, 1.1)
    X = [
        data(a) for a in numpy.logspace(-round(numpy.log10(K)), 0, R)
        for _ in range(NITER)
    ]
    refs = [entropy(p1) for _, p1 in X]

    fonts = figs.TalkFonts
    fonts['axes.titlesize'] = 'small'
    with mpl.style.context([figs.Custom, figs.FigSizeL, fonts],
                           after_reset=True):
        fig, axs = plt.subplots(1, 4, figsize=figs.fig_size(5, 3))

        t0 = time()
        ar2 = [(entropy(p1), entropy(x)) for x, p1 in X]
        print('time0: ', time() - t0)
        ar2 = numpy.array(list(zip(*ar2))) / numpy.log(K)
        with figs.axes(ax=axs[0],
                       title='scipy.stats.entropy(counts)',
                       xlim=ylim,
                       ylim=ylim,
                       ylabel='entropy estimate') as ax:
            plot_errorbar(ax, ar2)

        t0 = time()
        ar0 = [(entropy(p1), *ndd.entropy(x, k=K, return_std=1))
               for x, p1 in X]
        ar0 = [x for x in ar0 if None not in x]
        print('time0: ', time() - t0)
        ar0 = numpy.array(list(zip(*ar0))) / numpy.log(K)
        with figs.axes(ax=axs[1],
                       title='ndd.entropy(counts, k=k)',
                       xlim=ylim,
                       ylim=ylim,
                       xlabel='true entropy',
                       y_invisible=True) as ax:
            plot_errorbar(ax, ar0, label='ndd.entropy')

        t0 = time()
        ar1 = [(entropy(p1), *ndd.entropy(x, return_std=1)) for x, p1 in X]
        ar1 = [x for x in ar1 if None not in x]
        print('time0: ', time() - t0)
        ar1 = numpy.array(list(zip(*ar1))) / numpy.log(K)
        with figs.axes(ax=axs[2],
                       title='ndd.entropy(counts)',
                       xlim=ylim,
                       ylim=ylim,
                       y_invisible=True) as ax:
            plot_errorbar(ax, ar1, label='ndd.entropy')

        t0 = time()

        ar3 = []
        for x, p1 in X:
            try:
                S, err = ndd.entropy(x,
                                     estimator='AsymptoticNSB',
                                     return_std=1)
            except NddError:
                pass
            else:
                if not numpy.isinf(err):
                    ar3.append((entropy(p1), S, err))
        ar3 = [x for x in ar3 if None not in x]
        print('time0: ', time() - t0)
        ar3 = numpy.array(list(zip(*ar3))) / numpy.log(K)
        with figs.axes(ax=axs[3],
                       title='ndd.entropy(counts)',
                       xlim=ylim,
                       ylim=ylim,
                       y_invisible=True) as ax:
            plot_errorbar(ax, ar3, label='ndd.entropy')

        # plt.tight_layout()
        plt.show()
        fig.savefig('fig1.pdf', format='pdf')

    # compute bias and std for k-guess
    ar0[1] = ar0[1] - ar0[0]
    ar1[1] = ar1[1] - ar1[0]
    print(numpy.mean(ar0[1]), numpy.std(ar0[1]))
    print(numpy.mean(ar1[1]), numpy.std(ar1[1]))
