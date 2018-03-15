# -*- coding: utf-8 -*-
from __future__ import print_function
from pkg_resources import parse_version

NAME = 'ndd'
NUMPY_MIN_VERSION = '1.9'
VERSION_FILE = 'ndd/version.json'
SETUP_REQUIRES = ['numpy']
INSTALL_REQUIRES = ['future', 'pytest', 'scipy']

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
        status['up_to_date'] = parse_version(
            numpy_version) >= parse_version(NUMPY_MIN_VERSION)
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
        raise KeyError("check version file: no version number")

def get_long_description():
    """Get the long description from the README file."""
    from os import path
    import codecs
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, 'README.rst'), encoding='utf-8') as _rf:
        return _rf.read()

# check numpy first
NUMPY_STATUS = get_numpy_status()
NUMPY_REQ_STR = "ndd requires NumPy >= %s" % NUMPY_MIN_VERSION
if NUMPY_STATUS['up_to_date'] is False:
    if NUMPY_STATUS['version']:
        raise ImportError(
            "Your installation of NumPy %s is out-of-date.\n%s"
            % (NUMPY_STATUS['version'], NUMPY_REQ_STR))
    else:
        raise ImportError(
            "NumPy is not installed.\n%s"
            % NUMPY_REQ_STR)

from numpy.distutils.core import setup # pylint: disable=wrong-import-position
#from setuptools import setup # pylint: disable=wrong-import-position
from numpy.distutils.core import Extension # pylint: disable=wrong-import-position

VERSION = get_version(VERSION_FILE)
LONG_DESCRIPTION = get_long_description()

_NSB = Extension(
    name='ndd._nsb',
    sources=['ndd/nsb.pyf',
             'ndd/exts/gamma.f90',
             'ndd/exts/quad.f90',
             'ndd/exts/estimators.f90'],
    #extra_f90_compile_args = ["-fopenmp"],
    #extra_link_args = ["-lgomp"],
)

setup(
    name=NAME,
    version=VERSION,
    description="Bayesian entropy estimation from discrete data",
    long_description=LONG_DESCRIPTION,
    author='Simone Marsili',
    author_email='simo.marsili@gmail.com',
    url='https://github.com/simomarsili/ndd',
    keywords='entropy estimation Bayes discrete_data',
    data_files=[(NAME, ['ndd/version.json'])],

    #py_modules=['ndd'],
    packages=['ndd'],
    ext_modules=[_NSB],

    entry_points={
        'console_scripts': [
            'ndd=ndd.entry:main'
        ]},

    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': ['pytest'],
        'docs': ['mkdocs']},
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
)
