# ndd

**ndd** is a Python/Fortran module for estimation of entropy and entropy-related quantities from discrete data.  
The main goal of **ndd** is to provide a simple and minimal Python interface to the Nemenman-Shafee-Bialek (NSB) algorithm,
a state of the art Bayesian algorithm for entropy estimation. In **ndd**, this interface is the ndd.entropy() function,
that takes as input an array of counts over a set of possible outcomes, and returns an entropy estimate:  
entropy_estimate = ndd.entropy(counts). 

# Obtaining the source

All **ndd** source code is hosted on Github. 
You can download the latest version of the code using [this link](https://github.com/simomarsili/ndd/archive/v0.1.8.zip). 

# Prerequisites

In order to compile **ndd**, you will need to have a **Fortran compiler** installed on your machine.   
If you are using Debian or a Debian derivative such as Ubuntu, you can install the gfortran compiler using the following command:

    sudo apt-get install gfortran

The compiling and linking of source files is handled by **Gnu Make**. 
If you are using Debian or a Debian derivative such as Ubuntu, you should find 4.1 already installed. 

Finally, you will need to install the **numpy** library: 
   
    sudo apt-get install python-numpy

that will provide also **f2py** (the Fortran to Python interface generator) - or install the full [SciPy stack](https://www.scipy.org/install.html).

# Install 

To install **ndd**, enter the root directory:
     
    cd ndd

and type make:

    make

This will build and install the ndd module in /usr/local . 
The install path can be specified on the make command line as an absolute path, e.g. : 

    make INSTALL_PATH=~/.local

will install the module in the .local dir in your home. 

# Testing

From the root directory of the project, type: 

    make test

# Basic usage 

  The ndd.entropy function takes as input a histogram (a list or a numpy array of integers representing counts) and returns a entropy estimate (in nats): 

    >>> counts
    [ 7  3  5  8  9  1  3  3  1  0  2  5  2 11  4 23  5  0  8  0]
    >>> import ndd
    >>> estimated_entropy = ndd.entropy(counts)
    >>> estimated_entropy
    2.623634344902917

  Compared to the standard, "plugin" estimator, the NSB estimator performs well in the undersampled regime
  (i.e. k >> n where k is the number of possible outcomes and n the number of samples).
  Check the [tutorial](https://github.com/simomarsili/ndd/blob/master/notebooks/ndd_tutorial.ipynb) for more info.

# Contributing

**ndd** is an OPEN Source Project so please help out by [reporting bugs](http://github.com/simomarsili/ndd/issues) or [forking and opening pull](https://github.com/simomarsili/ndd) requests when possible.

# LICENSE (BSD 3 clause)

Copyright (c) 2016, Simone Marsili.   
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

