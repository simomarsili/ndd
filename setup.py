# -*- coding: utf-8 -*-
from os import path
import codecs
import re
#from setuptools import setup
import pip

setup_requires = ['numpy']
install_requires=['future', 'pytest', 'scipy']
# install numpy via pip
pip.main(['install'] + setup_requires)
setup_requires = []


from numpy.distutils.core import setup
from numpy.distutils.core import Extension

# get the long description from the README file
here = path.abspath(path.dirname(__file__))
with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get version number(s)
VERSIONFILE = "version.py"
version_string = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, version_string, re.M)
if mo:
    VERSION = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

nddf = Extension(
    name = 'nddf',
    sources = ['exts/ndd.pyf','exts/gamma.f90','exts/quad.f90',
               'exts/ndd.f90'],
    #extra_f90_compile_args = ["-fopenmp"],
    #extra_link_args = ["-lgomp"],
)

setup(
    name='ndd',
    version=VERSION,
    description="Estimates of entropy and entropy-related quantities from discrete data",
    long_description=long_description,
    author='Simone Marsili',
    author_email='simo.marsili@gmail.com',
    url='https://github.com/pypa/sampleproject',
    #packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    py_modules        = ['ndd'],
    ext_modules       = [nddf],
    setup_requires = setup_requires,
    install_requires = install_requires,
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    #extras_require={
    #    'dev': ['check-manifest'],
    #    'test': ['coverage'],
    #},
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
