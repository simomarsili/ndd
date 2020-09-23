# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Entry points. """
import json
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import ndd

available_estimators = ', '.join(ndd.entropy_estimators.keys())


def parse_args():
    """Parse command line options."""
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-k',
                        '--alphabet_size',
                        type=float,
                        help='alphabet size (override value in input file)',
                        default=None)
    parser.add_argument('-e',
                        '--estimator',
                        type=str,
                        help='Use a specific estimator (available: '
                        '%s)' % available_estimators,
                        default=None)
    parser.add_argument('-a',
                        '--alpha',
                        type=float,
                        help='concentration parameter/pseudocounts',
                        default=None)
    return parser.parse_args()


def main():
    """ndd script."""

    args = parse_args()

    data = json.load(sys.stdin)

    k = args.alphabet_size
    zk = None
    if isinstance(data, dict) and 'nk' in data:
        nk = data['nk']
        k = k or data.get('k', None)
        zk = data.get('zk', None)
    else:
        # list of integers counts or mapping to ints
        nk = data

    if args.estimator is not None:
        try:
            algorithm = ndd.entropy_estimators[args.estimator]
        except KeyError:
            raise ValueError('Invalid estimator. Available estimators:\n'
                             '%s' % available_estimators)
        if args.alpha is not None:  # use alpha only if the estimator is given
            try:
                estimator = algorithm(alpha=args.alpha)
            except TypeError:
                estimator = algorithm()
    else:
        estimator = None

    _ = ndd.entropy(nk, k=k, zk=zk, estimator=estimator)
    json.dump(ndd.entropy.info, sys.stdout, indent=4)


if __name__ == '__main__':
    main()
