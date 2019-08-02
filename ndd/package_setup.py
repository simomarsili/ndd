# -*- coding: utf-8 -*-
"""Pre-import project setup."""
import os
import platform

SEP = os.path.sep
PATHSEP = os.pathsep
PLATFORM = platform.system()


def add_dir_to_path(a):
    """Append a dir to PATH if exists else return None"""
    if os.path.isdir(a):
        os.environ['PATH'] += PATHSEP + a
    else:
        a = None
    return a


def add_libs_dir():
    """In Windows systems, add the .libs to PATH env variable (if exists)."""
    libs = os.path.join(package_path, '.libs')
    return add_dir_to_path(libs)


def subclasses(cls):
    """Return a dict name -> class for all subclasses of class `cls`."""
    return {sc.__name__: sc for sc in cls.__subclasses__()}


package_path = os.path.dirname(os.path.abspath(__file__))
package_name = package_path.split(SEP)[-1]

if PLATFORM == 'Windows':
    libs_dir = add_libs_dir()
