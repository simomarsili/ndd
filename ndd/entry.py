# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)


def main():
    import sys
    import json
    import ndd
    """
    with open(sys.stdin, 'r') as f:
        data = json.load(f)
        print(data)
    """
    # load data from json file
    histogram = json.load(sys.stdin)
    entropy, error = ndd.entropy(list(histogram.values()), return_std=True)
    json.dump(
        {'entropy': entropy,
         'error': error
        },
        sys.stdout,
        indent=4
    )


if __name__ == '__main__':
    main()
