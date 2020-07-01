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

N = 1000
P = 1000000
ALPHA = 0.001
R = 50

numpy.random.seed(1234)


def random_counts(a=None):
    """
    Return the entropy of a random probability vector p ~ Dirichlet(a) and
    random counts drawn from Multinomial(p).

    Args:
      a (float): Scalar concentration parameter [a]*P.

    Returns:
      array-like: array of counts.
      float: entropy of the random probability vector.

    """
    if a is None:
        a0 = -numpy.log10(P)
        a1 = 0
        a = a0 + numpy.random.rand() * (a1 - a0)
        a = 10**a
    while 1:
        try:
            pp = dirichlet([a] * P)
            counts = multinomial(N, pp)
        except ValueError:
            pass
        else:
            break
    counts = counts[numpy.nonzero(counts)]
    return counts, entropy(pp) / numpy.log(P)


def entropy_estimate(a=None, r=1):
    """
    Return entropy ref value, estimates and err estimates for the
    plugin/nsb/auto estimators.
    """
    if r and r >= 2:
        true = []
        ests = []
        err = []
        bounds = []
        for _ in range(r):
            S0, x, y, b = entropy_estimate(a=a)
            true.append(S0)
            ests.append(x)
            err.append(y)
            bounds.append(b)
        return (numpy.array(true), numpy.array(ests).T, numpy.array(err).T,
                numpy.array(bounds))
    c, S0 = random_counts(a=a)
    S = [None] * 3
    err = [None] * 3
    S[0], err[0] = entropy(c) / numpy.log(P), None
    S[1], err[1] = ndd.entropy(c, return_std=1) / numpy.log(P)
    entropy_estimate.bounded = ndd.entropy.bounded
    S[2], err[2] = ndd.entropy(c, k=P, return_std=1) / numpy.log(P)
    return S0, tuple(S), tuple(err), entropy_estimate.bounded


def plot_errorbar(axes, x, y, err, c=None):
    # pylint: disable=invalid-unary-operand-type
    """Plotter"""
    kwargs = {
        'alpha': 0.5,
        'elinewidth': 1,
        'capsize': 3,
        'errorevery': 1,
        'ls': '',
    }

    if c is None:
        c = list(range(len(x)))

    if None in err:
        axes.scatter(x[c], y[c], zorder=1, marker='o', alpha=0.5)
        axes.scatter(x[~c], y[~c], zorder=1, marker='x', alpha=0.5, c='teal')
        return axes

    axes.errorbar(x[c], y[c], **kwargs, yerr=err[c], zorder=1, marker='o')
    axes.errorbar(x[~c],
                  y[~c],
                  **kwargs,
                  yerr=err[~c],
                  zorder=1,
                  marker='x',
                  c='teal')
    axes.plot(x, x, '--', lw=1, zorder=2, c='black', alpha=0.5)
    return axes


if __name__ == '__main__':
    ylim = (-0.2, 1.2)

    xs, ys, errs, bounded = entropy_estimate(r=R)

    titles = [
        'scipy.stats.entropy(counts)', 'ndd.entropy(counts)',
        'ndd.entropy(counts, k=k)'
    ]
    xlabels = [None, 'entropy', None]
    ylabels = ['entropy estimate'] + [None] * 2

    fonts = figs.PosterFonts
    fonts['axes.titlesize'] = 'small'
    with mpl.style.context([figs.Custom, figs.FigSizeL, fonts],
                           after_reset=True):
        fig, axs = plt.subplots(1, 3, figsize=figs.fig_size(16, 6))

        for j, title in enumerate(titles):
            with figs.axes(
                    ax=axs[j],
                    title=title,
                    xlim=ylim,
                    ylim=ylim,
                    xlabel=xlabels[j],
                    ylabel=ylabels[j],
                    # x_invisible=j != 1,
                    y_invisible=j != 0) as ax:
                plot_errorbar(ax, xs, ys[j], errs[j], c=bounded)
        plt.show()
        fig.savefig('bars.svg', format='svg')
