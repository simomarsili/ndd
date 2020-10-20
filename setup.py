# -*- coding: utf-8 -*-
"""Setup module."""
# pylint: disable=wrong-import-position
from __future__ import print_function

import codecs
import platform
from os import path

# import setuptools before imporing setup from numpy.distutils.core
# https://stackoverflow.com/a/55358607
from pkg_resources import parse_version
from setuptools import find_packages

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


def get_package_name():
    'The top-level package name.'
    top_level_packages = [
        p for p in find_packages(exclude=['tests']) if '.' not in p
    ]
    if len(top_level_packages) != 1:
        raise ValueError('Project must contain a single top-level package.')
    return top_level_packages[0]


def get_long_description(readme):
    """Get the long description from the README file."""
    with codecs.open(readme, encoding='utf-8') as _rf:
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

base_dir = path.abspath(path.dirname(__file__))
readme_file = path.join(base_dir, 'README.rst')

# single top-level package
package_name = get_package_name()

# get project info from the package.py module of the top-level package
project_info = {}
with open(path.join(package_name, 'package.py')) as fp:
    exec(fp.read(), project_info)  # pylint: disable=exec-used

project_name = project_info['__title__']
project_version = project_info['__version__']

long_description = get_long_description(readme_file)

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
    name=project_name,
    version=project_version,
    description='Bayesian entropy estimation from discrete data',
    long_description=long_description,
    # long_description_content_type="text/markdown",
    author=project_info.get('__author__'),
    author_email=project_info.get('__email__'),
    url=project_info.get('__url__'),
    packages=[package_name],
    package_data={'': ['LICENSE.txt', 'README.rst', 'requirements.txt']},
    ext_modules=[FNSB],
    entry_points={'console_scripts': ['ndd=ndd.entry:main']},
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
