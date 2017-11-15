# -*- coding: utf-8 -*-
from __future__ import print_function
from pkg_resources import parse_version

NUMPY_MIN_VERSION = '1.8'
VERSION_FILE = 'version.json'
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
    """ Retrieve ndd version number."""
    import json
    with open(source, 'r') as _vf:
        version_data = json.load(_vf)
    try:
        return version_data['number']
    except KeyError:
        raise KeyError("check version file: no version number")

def get_long_description():
    """Get the long description from the README file."""
    from os import path
    import codecs
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
        return f.read()

# check numpy first
numpy_status = get_numpy_status()
numpy_req_str = "ndd requires NumPy >= %s" % NUMPY_MIN_VERSION
if numpy_status['up_to_date'] is False:
    if numpy_status['version']:
        raise ImportError(
            "Your installation of NumPy %s is out-of-date.\n%s"
            % (numpy_status['version'], numpy_req_str))
    else:
        raise ImportError(
            "NumPy is not installed.\n%s"
            % numpy_req_str)

from numpy.distutils.core import setup
from numpy.distutils.core import Extension

# ndd version
ndd_version = get_version(VERSION_FILE)

long_description = get_long_description()

nddf = Extension(
    name = 'nddf',
    sources = ['exts/ndd.pyf','exts/gamma.f90','exts/quad.f90',
               'exts/ndd.f90'],
    #extra_f90_compile_args = ["-fopenmp"],
    #extra_link_args = ["-lgomp"],
)

setup(
    name='ndd',
    version=ndd_version,
    description="Entropy from discrete data",
    long_description=long_description,
    author='Simone Marsili',
    author_email='simo.marsili@gmail.com',
    url='https://github.com/simomarsili/ndd',
    keywords='entropy estimation Bayes discrete_data',
    py_modules=['ndd'],
    ext_modules=[nddf],
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
