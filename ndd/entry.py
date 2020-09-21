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
    if isinstance(data, dict) and 'nk' in data:
        nk = data['nk']
        k = data.get('k', None)
        zk = data.get('zk', None)
    else:
        # list of integer counts or histogram dict
        nk = data

    _ = ndd.entropy(nk, k=k, zk=zk)
    json.dump(ndd.entropy.info, sys.stdout, indent=4)


if __name__ == '__main__':
    main()
