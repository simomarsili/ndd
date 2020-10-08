# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""
Entropy estimate from frequency counts data.

The input file must be a JSON file containing either:
* a list of integer counts:
  [1, 2, 3, 4]
* a dictionary mapping class labels to integer counts:
  {"a": 1, "b": 2, "c": 3, "d": 4}
* a dictionary storing counts either as a list/dict and
  (optionally) the alphabet size:
  {
      "nk": [1, 2, 3, 4],
      "k": 100
  }

"""
import json
import sys
from argparse import ArgumentParser, FileType, RawDescriptionHelpFormatter

import ndd

available_estimators = ', '.join(ndd.entropy_estimators.keys())


def parse_args():
    """Parse command line options."""
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                            description=__doc__)
    parser.add_argument('-i',
                        '--input_file',
                        nargs='?',
                        type=FileType('r'),
                        default=sys.stdin)
    parser.add_argument('-o',
                        '--output_file',
                        nargs='?',
                        type=FileType('w'),
                        default=sys.stdout)
    parser.add_argument('-k',
                        '--alphabet_size',
                        type=float,
                        help='alphabet size (override value in input file).',
                        default=None)
    parser.add_argument('-e',
                        '--estimator',
                        type=str,
                        help='Use a specific estimator (available: '
                        '%s.)' % available_estimators,
                        default=None)
    parser.add_argument('-a',
                        '--alpha',
                        type=float,
                        help='Concentration parameter/pseudocounts '
                        '(only if a specific estimator is passed as '
                        'an argument).',
                        default=None)
    return parser.parse_args()


def compute_estimate(data, k=None, estimator=None, alpha=None):
    """Entropy estimation given data."""

    zk = None
    if isinstance(data, dict) and 'nk' in data:
        nk = data['nk']
        k = k or data.get('k', None)
        zk = data.get('zk', None)
    else:
        # list of integers counts or mapping to ints
        nk = data

    if estimator is not None:
        try:
            algorithm = ndd.entropy_estimators[estimator]
        except KeyError:
            raise ValueError('Invalid estimator. Available estimators:\n'
                             '%s' % available_estimators)
        if alpha is not None:  # use alpha only if the estimator is given
            try:
                estimator = algorithm(alpha=alpha)
            except TypeError:
                estimator = algorithm()
    else:
        estimator = None

    if zk:
        _ = ndd.entropy((nk, zk), k=k, estimator=estimator)
    else:
        _ = ndd.entropy(nk, k=k, estimator=estimator)

    return ndd.entropy.info


def main():
    """ndd script."""

    args = parse_args()
    try:
        data = json.load(args.input_file)
    except json.decoder.JSONDecodeError:
        raise ndd.exceptions.NddError('Input file must be a valid JSON file.')
    result = compute_estimate(data,
                              k=args.alphabet_size,
                              estimator=args.estimator,
                              alpha=args.alpha)
    json.dump(result, args.output_file, indent=4)


if __name__ == '__main__':
    main()
