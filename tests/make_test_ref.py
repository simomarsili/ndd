# -*- coding: utf-8 -*-
"""Prepare test .json files."""
# pylint: disable=missing-docstring
import itertools
import json
import sys

import numpy
import numpy.random as random

import ndd

SEED = 123


def random_counts(n=None, k=None, alpha=None):
    """Sample from random multinomial."""
    random.seed(SEED)
    pp = random.dirichlet([alpha] * k)
    return random.multinomial(n, pp)


def estimator_prms():
    return [
        {},  # NSB
        {
            'alpha': 1.0
        },  # Dirichlet
        {
            'alpha': 0.0,
            'plugin': 1
        },  # ML
        {
            'alpha': 1.0,
            'plugin': 1
        },  # pseudo
    ]


def counts_prms():
    ns = [int(x) for x in numpy.logspace(1, 3, num=3)]
    ks = [int(x) for x in numpy.logspace(1, 3, num=3)]
    alphas = numpy.logspace(-2, 1, num=4)
    # list of combinations of parameter values
    return ({
        'n': n,
        'k': k,
        'alpha': alpha
    } for n, k, alpha in itertools.product(ns, ks, alphas))


def cases():
    for prms in counts_prms():
        counts = random_counts(**prms)
        for kwargs in estimator_prms():
            yield (counts, prms, kwargs)


def main():
    """Main function."""
    results = []
    for counts, _, kwargs in cases():
        result = ndd.entropy(counts, **kwargs, k=len(counts))
        results.append(result)
    json.dump(results, sys.stdout)


if __name__ == '__main__':
    main()
