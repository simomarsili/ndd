# -*- coding: utf-8 -*-
"""Utils functions to store failing cases.

A dir "failed" will be created if not existing

The PYTHONHASHSEED environment variable must be set in advance.

"""
import json
import logging
import os
from functools import wraps
from inspect import isclass
from pathlib import Path

import numpy

import ndd

logger = logging.getLogger(__name__)

if 'PYTHONHASHSEED' not in os.environ:  # for reproducible hash values
    logger.warning('WARNING: PYTHONHASHSEED is not set. '
                   'No file will be written.')


class NpEncoder(json.JSONEncoder):
    """Convert objects to JSON serilizable."""

    def default(self, obj):  # pylint: disable=arguments-differ, method-hidden
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, (ndd.EntropyEstimator, ndd.DivergenceEstimator)):
            return obj.__class__.__name__
        if isclass(obj):
            return obj.__name__
        return super(NpEncoder, self).default(obj)


def default_unexpected(x):
    """Define unexpected values."""
    large = 1.e3
    small = -numpy.float('inf')
    return (x is None) or x > large or x < small or numpy.isnan(x)


def dump_on_fail(fail_dir='failed', unexpected=default_unexpected):
    """Dump args for unexpected results to `fp`.

    Parameters
    ----------
    fail_dir : path or str
        Save failing cases args to `fail_dir`.
    unexpected : callable
        Return True if input value is not expected.

    """

    fail_dir = Path(fail_dir)

    def has_unexpected_values(result):
        result = numpy.array(result, copy=False, ndmin=1)
        return numpy.any([unexpected(r) for r in result])

    def decorate(func):
        @wraps(func)
        def dump_args_to_file(*args, **kwargs):
            result = func(*args, **kwargs)
            if has_unexpected_values(result):
                case = {
                    'f': func.__qualname__,
                    'args': args,
                    'kwargs': kwargs,
                    'result': result
                }
                code = str(hash(tuple(case)))
                filename = fail_dir / code
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as fp:
                    print(json.dumps(case, cls=NpEncoder), file=fp)
            return result

        dump_args_to_file.original_func = func
        return dump_args_to_file

    return decorate


def run_case(path):
    """Run case from file."""
    if hasattr(path, 'read'):
        case = json.load(path)
    else:
        with open(path) as fp:
            case = json.load(fp)

    func = getattr(ndd, case['f'])
    args = case['args']
    kwargs = case['kwargs']
    result = func.original_func(*args, **kwargs)
    return (case, result)


def run_failed_cases(fail_dir='failed'):
    """Run all failed cased."""
    for path in Path(fail_dir).glob('**/*'):
        result = run_case(path)[1]
        print(str(path), result)


if __name__ == '__main__':
    run_failed_cases()
