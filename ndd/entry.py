# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Entry points. """
import json
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import ndd

available_estimators = ', '.join(ndd.entropy_estimators.keys())


def parse_args():
    """Parse command line options."""
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-i',
                        '--input_file',
                        type=str,
                        help='JSON input file; defaults to stdin.',
                        default=None)
    parser.add_argument('-o',
                        '--output_file',
                        type=str,
                        help='Output file; defaults to stdout.',
                        default=None)
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

    _ = ndd.entropy(nk, k=k, zk=zk, estimator=estimator)

    return ndd.entropy.info


def main():
    """ndd script."""

    args = parse_args()

    if args.input_file is None:
        data = json.load(sys.stdin)
    else:
        with open(Path(args.input_file), 'r') as fp:
            data = json.load(fp)

    info = compute_estimate(data,
                            k=args.alphabet_size,
                            estimator=args.estimator,
                            alpha=args.alpha)

    if args.output_file is None:
        json.dump(info, sys.stdout, indent=4)
    else:
        with open(Path(args.output_file), 'w') as fp:
            json.dump(info, fp, indent=4)


if __name__ == '__main__':
    main()
