# -*- coding: utf-8 -*-
"""Pre-import package setup."""
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


def subclasses(cls, abstract=False, private=False):
    """Return the subclasses of class `cls` as a dict.

    If abstract, include classes with abstract methods.
    If private, include private classes.
    """
    return {
        sc.__name__: sc
        for sc in cls.__subclasses__()
        if (abstract or not sc.__abstractmethods__) and (
            private or sc.__name__[0] != '_')
    }


package_path = os.path.dirname(os.path.abspath(__file__))
package_name = package_path.split(SEP)[-1]

if PLATFORM == 'Windows':
    libs_dir = add_libs_dir()
