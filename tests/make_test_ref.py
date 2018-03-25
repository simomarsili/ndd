import sys
import numpy
import numpy.random as random
import itertools
import json
import ndd

def random_counts(n=None, k=None, alpha=None):
    random.seed(123)
    pp = random.dirichlet([alpha]*k)
    return random.multinomial(n, pp)

# a test should include:
# name, input setting, ndd setting, result

methods = ['NSB', 'Dirichlet', 'ML', 'pseudo']

ndd_kwargs = {
    'NSB' : {},
    'Dirichlet' : {'alpha':1.0},
    'ML' : {'alpha':0.0, 'plugin':1},
    'pseudo' : {'alpha':1.0, 'plugin':1},
}

n_array = [int(x) for x in numpy.logspace(1, 3, num=3)]
k_array = [int(x) for x in numpy.logspace(1, 3, num=3)]
a_array = numpy.logspace(-2, 1, num=4)

# list of combinations of parameter values
settings = [{'n':n, 'k':k, 'alpha':alpha}
            for n, k, alpha in itertools.product(n_array, k_array, a_array)]

test_methods = {}
for method in methods:
    tests = []
    kwargs = ndd_kwargs[method]
    for setting in settings:
        kwargs['k'] = setting['k']
        counts = random_counts(**setting)
        result = ndd.entropy(counts, **kwargs)
        ndd_in = {'k':setting['k']}
        for key, val in kwargs.items():
            ndd_in[key] = val
        test = (setting, ndd_in, result)
        tests.append(test)
    test_methods[method] = tests
json.dump(test_methods, sys.stdout)

