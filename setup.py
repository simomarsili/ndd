# -*- coding: utf-8 -*-
"""Setup module."""
# pylint: disable=wrong-import-position
from __future__ import print_function

import platform

# import setuptools before imporing setup from numpy.distutils.core
# https://stackoverflow.com/a/55358607
from pkg_resources import parse_version

NAME = 'ndd'
NUMPY_MIN_VERSION = '1.13'
PACKAGE_FILE = 'package.json'
SETUP_REQUIRES = ['numpy>=' + NUMPY_MIN_VERSION]
INSTALL_REQUIRES = []
EXTRAS_REQUIRES = {'test': ['pytest']}
PLATFORM = platform.system()


def get_numpy_status():
    """
    Returns a dictionary containing a boolean specifying whether NumPy
    is up-to-date, along with the version string (empty string if
    not installed).
    From pymc: https://raw.githubusercontent.com/pymc-devs/pymc/master/setup.py
    """
    status = {}
    try:
        import numpy
        numpy_version = numpy.__version__
        status['up_to_date'] = parse_version(numpy_version) >= parse_version(
            NUMPY_MIN_VERSION)
        status['version'] = numpy_version
    except ImportError:
        status['up_to_date'] = False
        status['version'] = ''
    return status


def get_version(source):
    """ Retrieve version number."""
    import json
    with open(source, 'r') as _vf:
        version_data = json.load(_vf)
    try:
        return version_data['version']
    except KeyError:
        raise KeyError('check version file: no version number')


def get_long_description():
    """Get the long description from the README file."""
    from os import path
    import codecs
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, 'README.rst'), encoding='utf-8') as _rf:
        return _rf.read()


# check numpy first
NUMPY_STATUS = get_numpy_status()
NUMPY_REQ_STR = "ndd requires NumPy >= %s. Run 'pip install -U numpy' " % NUMPY_MIN_VERSION
if NUMPY_STATUS['up_to_date'] is False:
    if NUMPY_STATUS['version']:
        raise ImportError('Your installation of NumPy %s is out-of-date.\n%s' %
                          (NUMPY_STATUS['version'], NUMPY_REQ_STR))
    raise ImportError('NumPy is not installed.\n%s' % NUMPY_REQ_STR)

from numpy.distutils.core import Extension  # isort:skip
from numpy.distutils.core import setup  # isort:skip

VERSION = get_version(PACKAGE_FILE)
LONG_DESCRIPTION = get_long_description()

FSOURCES = [
    'ndd/nsb.pyf', 'ndd/exts/gamma.f90', 'ndd/exts/quad.f90',
    'ndd/exts/estimators.f90'
]
EXT_NAME = 'ndd.fnsb'


def extension_args():
    """Extension object parameters."""
    args = {'name': EXT_NAME, 'sources': FSOURCES}
    platform_specific = {
        'Darwin': {
            'extra_link_args': ['-undefined', 'dynamic_lookup']
        },
    }
    args.update(platform_specific.get(PLATFORM, dict()))
    return args


FNSB = Extension(**extension_args())

setup(
    name=NAME,
    version=VERSION,
    description='Bayesian entropy estimation from discrete data',
    long_description=LONG_DESCRIPTION,
    # long_description_content_type="text/markdown",
    author='Simone Marsili',
    author_email='simo.marsili@gmail.com',
    url='https://github.com/simomarsili/ndd',
    keywords='entropy estimation Bayes discrete_data',
    data_files=[(NAME, ['package.json'])],
    packages=['ndd'],
    package_data={'': ['LICENSE.txt', 'README.rst', 'requirements.txt']},
    ext_modules=[FNSB],
    python_requires='>=3.5',
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require={'test': ['pytest']},
    license='BSD 3-Clause',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
