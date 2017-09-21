import sys
import numpy
import numpy.random
import itertools
import json
import ndd

As = numpy.logspace(-2, 1, num=4)
Ms = [int(x) for x in numpy.logspace(1, 3, num=3)]
Ns = [int(x) for x in numpy.logspace(1, 3, num=3)]

tests = list(itertools.product(As, Ms, Ns))

for t, test in enumerate(tests):
    test = list(test)
    a, m, n = test
    numpy.random.seed(123)
    pp = numpy.random.dirichlet([a]*m)
    data = numpy.random.multinomial(n, pp)
    test.append(ndd.entropy(data))
    tests[t] = test
    

json.dump(tests, sys.stdout)

