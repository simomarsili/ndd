# -*- coding: utf-8 -*-
"""Fast test."""
import numpy

import ndd

a = [7, 3, 5, 8, 9, 1, 3, 3, 1, 0, 2, 5, 2, 11, 4, 23, 5, 0, 8, 0]
h = ndd.entropy(a, k=len(a))
# href = 2.623634344888532
# href = 2.623634344902917
href = 2.6192575031125798
absolute_error = numpy.abs(h - href)
relative_error = absolute_error / href
# smallest positive number in single precision
eps = numpy.finfo(numpy.float32).eps
try:
    assert absolute_error < eps
except AssertionError:
    raise AssertionError('estimate %r /= %r' % (h, href))
else:
    print('%r. Abs. error is %r. Test ok!' % (h, absolute_error))
