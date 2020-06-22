# -*- coding: utf-8 -*-
"""Prepare test .json files."""
# pylint: disable=missing-docstring
import itertools
import json
import sys

import numpy
import numpy.random as random
import pytest

import ndd
from ndd.estimators import NSB, Plugin

SEED = 123


def approx(a, rel=1.e-5, abs=1.e-3):  # pylint: disable=redefined-builtin
    """Wrap pytest approx. Parms values from numpy.isclose()"""
    return pytest.approx(a, rel=rel, abs=abs)


def random_counts(n=None, k=None, alpha=None):
    """Sample from random multinomial."""
    random.seed(SEED)
    pp = random.dirichlet([alpha] * k)
    return random.multinomial(n, pp)


def estimator_prms():
    return [
        {},  # NSB
        {
            'estimator': NSB(alpha=1.0),
        },  # Wolpert-Wolf
        {
            'estimator': Plugin(),
        },  # plugin estimator
        {
            'estimator': Plugin(alpha=1.0),
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
        result = ndd.entropy(counts, k=len(counts), **kwargs)
        results.append(result)
    json.dump(results, sys.stdout)


if __name__ == '__main__':
    main()
