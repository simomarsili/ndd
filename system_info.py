# -*- coding: utf-8 -*-
"""System info"""

import platform
import subprocess
import sys

import numpy


class SystemInfo:
    """Collect system infos.

    Infos are defined as object properties.
    """

    @classmethod
    def attrs(cls):
        """Available system infos."""
        return [p for p in dir(cls) if isinstance(getattr(cls, p), property)]

    def __init__(self):
        # collect system infos
        self.infos = {a: getattr(self, a) for a in self.attrs()}

    @property
    def platform(self):
        """Info on the underlying platform."""
        return platform.platform()

    @property
    def architecture(self):
        """System architecture."""
        is_64bits = sys.maxsize > 2**32
        arch = '64bits' if is_64bits else '32bits'
        return arch

    @property
    def python(self):
        """Python version."""
        return sys.version

    @property
    def numpy(self):
        """Numpy version."""
        return numpy.__version__

    @property
    def gfortran(self):
        """gfortran version."""
        return subprocess.run(['gfortran', '-v'],
                              stderr=subprocess.PIPE).stderr.decode()

    def __call__(self):
        return self.infos

    def __repr__(self):
        fmt = '\n'.join(['%s'] * 3 + ['\n'])
        return ''.join([
            fmt % (attr, '=' * len(attr), value)
            for attr, value in self().items()
        ])


if __name__ == '__main__':
    # print out system info
    print(SystemInfo())
