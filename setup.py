# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
from pkg_resources import parse_version

dist = sys.argv[1]
numpy_min_version = '1.8'
version_file = 'version.py'
setup_requires = ['numpy']
install_requires=['future', 'pytest', 'scipy']

def get_numpy_status():
    """
    Returns a dictionary containing a boolean specifying whether NumPy
    is up-to-date, along with the version string (empty string if
    not installed).
    From pymc: https://raw.githubusercontent.com/pymc-devs/pymc/master/setup.py
    """
    numpy_status = {}
    try:
        import numpy
        numpy_version = numpy.__version__
        numpy_status['up_to_date'] = parse_version(
            numpy_version) >= parse_version(numpy_min_version)
        numpy_status['version'] = numpy_version
    except ImportError:
        numpy_status['up_to_date'] = False
        numpy_status['version'] = ""
    return numpy_status

def get_ndd_version(vfile):
    """ Retrieve ndd version number."""
    import re
    version_string = open(vfile, "rt").read()
    vre = r"^__version__ = ['\"]([^'\"]*)['\"]"
    vmo = re.search(vre, version_string, re.M)
    if vmo:
        return vmo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (vfile,))

def get_long_description():
    """Get the long description from the README file."""
    from os import path
    import codecs
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
        return f.read()

# check numpy first
numpy_status = get_numpy_status()
numpy_req_str = "ndd requires NumPy >= %s" % numpy_min_version
if numpy_status['up_to_date'] is False:
    if numpy_status['version']:
        raise ImportError(
            "Your installation of NumPy %s is out-of-date.\n%s"
            % (numpy_status['version'], numpy_req_str))
    else:
        raise ImportError(
            "NumPy is not installed.\n%s"
            % numpy_req_str)

#install numpy via pip
#import pip
#pip.main(['install'] + setup_requires)
#setup_requires = []
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

# ndd version
ndd_version = get_ndd_version(version_file)

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
    setup_requires=setup_requires,
    install_requires=install_requires,
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
