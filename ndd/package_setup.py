# -*- coding: utf-8 -*-
"""Pre-import project setup."""
import os

SEP = os.path.sep
PATHSEP = os.pathsep


def add_libs_dir():
    """Add the .libs to PATH env variable (if exists)."""
    libs = os.path.join(package_path, '.libs')
    if os.path.isdir(libs):
        print('is dir')
        os.environ['PATH'] += PATHSEP + libs
    else:
        print('is not dir')
        libs = None
    return libs


def subclasses(cls):
    """Return a dict name -> class for all subclasses of class `cls`."""
    return {sc.__name__: sc for sc in cls.__subclasses__()}


package_path = os.path.dirname(os.path.abspath(__file__))
package_name = package_path.split(SEP)[-1]

libs_dir = add_libs_dir()
if libs_dir:
    print('%r added to PATH' % libs_dir)
