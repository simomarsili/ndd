# -*- coding: utf-8 -*-
"""Prepare test .json files."""
import itertools
import json
import sys

import numpy
import numpy.random as random

import ndd


def random_counts(n=None, k=None, alpha=None):
    """Sample from random multinomial."""
    random.seed(123)
    pp = random.dirichlet([alpha] * k)
    return random.multinomial(n, pp)


# a test should include:
# input setting, ndd setting, result
prms = [
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

n_vals = [int(x) for x in numpy.logspace(1, 3, num=3)]
k_vals = [int(x) for x in numpy.logspace(1, 3, num=3)]
a_vals = numpy.logspace(-2, 1, num=4)

# list of combinations of parameter values
settings = ({
    'n': n,
    'k': k,
    'alpha': alpha
} for n, k, alpha in itertools.product(n_vals, k_vals, a_vals))


def main():
    """Main function."""
    tests = []
    for setting in settings:
        k = setting['k']
        counts = random_counts(**setting)
        for kwargs in prms:
            # n. of classes is known
            result = ndd.entropy(counts, **kwargs, k=k)
            test_case = (setting, kwargs, result)
            tests.append(test_case)
    json.dump(tests, sys.stdout)


if __name__ == '__main__':
    main()
