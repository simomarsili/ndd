# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Entry points. """


def main():
    """ndd script."""
    import sys
    import json
    import ndd
    # load data from json file
    data = json.load(sys.stdin)

    k = None
    if isinstance(data, dict) and 'counts' in data:
        counts = data['counts']
        k = data.get('k', None)
    else:
        # list of integer counts or an histogram dict
        counts = data

    _ = ndd.entropy(counts, k=k)
    json.dump(ndd.entropy.info, sys.stdout, indent=4)


if __name__ == '__main__':
    main()
