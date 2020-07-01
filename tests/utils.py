# -*- coding: utf-8 -*-
"""Utils for tests."""
import os


def tests_dir():
    """Return None if no tests dir."""
    cwd = os.getcwd()
    basename = os.path.basename(cwd)
    if basename == 'tests':
        return cwd
    tdir = os.path.join(cwd, 'tests')
    if os.path.exists(tdir):
        return tdir
    return None
