# -*- coding: utf-8 -*-
"""Utils functions."""
import sys
from functools import wraps

import numpy


def unexpected_value(x):
    """Define supspicious values."""
    return (x is None) or x > 1.e4 or x == 0 or numpy.isnan(x)


def dump_on_fail(fp=sys.stdout):
    """Dump args for unexpected results to `fp`.

    Parameters
    ----------
    fp : path or file-like

    """
    # pylint:disable=unused-variable

    if not hasattr(fp, 'write'):
        fp = open(fp, 'w')

    def unexpected(result):
        result = numpy.array(result, copy=False, ndmin=1)
        return numpy.any(unexpected_value(result))

    def decorate(func):
        @wraps(func)
        def dump_args_to_file(*args, **kwargs):
            result = func(*args, **kwargs)
            if unexpected(result):
                print(func.__name__, *args, **kwargs, file=fp)


def delimited_to_camelcase(string, d='_', remove=None):
    """Convert string from delimiter_separated to CamelCase."""
    if d not in string:  # no delimiter
        if string[0].isupper():
            return string
        return string.title()
    string = string.title()
    if remove:
        string = string.replace(remove.lower().title(), '')
    return string.replace(d, '')  # TODO!!! use remove


def camelcase_to_delimited(string, d='_', remove=None):
    """Convert string from CamelCase to delimiter_separated."""
    result = []
    for i, c in enumerate(string):
        if c.isupper():
            if i > 0:
                result.append(d)
        result.append(c.lower())
    result = ''.join(result)
    if remove and remove in result:
        remove = remove.lower()
        result = d.join([x for x in result.split(d) if x != remove])
    return result


def as_class_name(*args, **kwargs):
    """Convert string into a CamelCase class name."""
    return delimited_to_camelcase(*args, **kwargs)


def register_class(cls, register):
    """Add a class to register."""
    register[cls.__name__] = cls
    return register
