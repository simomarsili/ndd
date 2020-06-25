====================================================
ndd - Bayesian entropy estimation from discrete data
====================================================
.. image:: https://badge.fury.io/py/ndd.svg
    :target: https://badge.fury.io/py/ndd
.. image:: https://travis-ci.com/simomarsili/ndd.svg?branch=master
    :target: https://travis-ci.com/simomarsili/ndd

The **ndd** package provides a simple Python interface to an efficient
implementation of the `Nemenman-Schafee-Bialek (NSB) algorithm
<https://arxiv.org/abs/physics/0108025>`_,
a parameter-free, Bayesian entropy estimator for discrete data.
The NSB algorithm allows entropy estimation when the number of samples is much
smaller than the number of classes with non-zero probability.


Basic usage
===========

The **entropy** function takes as input a vector of **frequency counts**
(the observed frequencies for a set of classes or states) and an **alphabet size**
(the number of classes with non-zero probability, including unobserved classes)
and returns an entropy estimate (in nats)::

  >>> import ndd
  >>> counts = [4, 12, 4, 5, 3, 1, 5, 1, 2, 2, 2, 2, 11, 3, 4, 12, 12, 1, 2]
  >>> ndd.entropy(counts, k=100)
  2.8060922529931225

The uncertainty in the entropy estimate can be quantified using the
posterior standard deviation (see Eq. 13 in `Archer 2013
<https://pillowlab.princeton.edu/pubs/Archer13_MIestim_Entropy.pdf>`_) ::

  >>> ndd.entropy(counts, k=100, return_std=True)
  (2.8060922529931225, 0.11945501149743358)

If the alphabet size is unknown or infinite and the `k` argument is omitted,
the `entropy` function will use an upper bound estimate for `k`, or switch to
the asymptotic NSB estimator
(Eqs. 29, 20 in
`Nemenman 2011 <https://nemenmanlab.org/~ilya/images/c/c1/Nemenman_2011b.pdf>`_
)
in the strongly undersampled regime (Eqs. 29, 20 in
`Nemenman 2011 <https://nemenmanlab.org/~ilya/images/c/c1/Nemenman_2011b.pdf>`_
) ::

  >>> ndd.entropy(counts)  # k is omitted
  2.8130746489179046
  >>> counts = [1]*100 + [2]*10  # undersampled distribution
  >>> entropy(counts)
  7.2072993808389789

Where to get it
===============
Install using pip::

  pip3 install -U ndd

or directly from sources in github for the latest version of the code::

  pip3 install git+https://github.com/simomarsili/ndd.git

In order to compile **ndd**, you will need **numpy** (>= 1.9) and a
**Fortran compiler**  installed on your machine.
If you are using Debian or a Debian derivative such as Ubuntu,
you can install the gfortran compiler using the following command::

  sudo apt-get install gfortran

On Windows, you can use the gfortran compiler from the
`MinGW-w64 <https://sourceforge.net/projects/mingw-w64/files>`_ project
(`direct link <https://sourceforge.net/projects/mingw-w64/files/latest/download>`_
to the installer).

If you don't have a Fortran compiler, install using the
`ndd python wheels <https://github.com/simomarsili/ndd-wheels>`_
with pre-compiled extensions. numpy >= 1.16 is needed.

Changes
=======

**v.1.9**
   Changed:

the ``entropy`` function takes frequency counts (multiplicities) as input via
   the ``zk`` optional argument

   if argument ``k`` is omitted, the ``entropy`` function will guess a
   reasonable alphabet size and select the best estimator for the sampling
   regime.

**v.1.8.3**
   Fixed:

   integration for huge cardinalities

**v1.8**
   Added:

   full Bayesian error estimate (from direct computation of the posterior
   variance of the entropy)

**v1.7**
   Changed:

   estimation is much faster (removed unnecessary checks on input counts)

   `entropy()` function needs cardinality `k` for the default (NSB) estimator

**v1.6.1**
   Changed:
   Fixed numerical integration for large alphabet sizes.

**v1.6**
   Changed:

   The signature of the *entropy* function has been changed to allow
   arbitrary entropy estimators. The new signature is::

     entropy(pk, k=None, estimator='NSB', return_std=False)

   The available estimators are::

     >>> import ndd
     >>> ndd.entropy_estimators
     ['Plugin', 'MillerMadow', 'NSB', 'AsymptoticNSB', 'Grassberger']

   Check the function docstring for details.

   Added:

   - *MillerMadow* estimator class
   - *AsymptoticNSB* estimator class
   - *Grassberger* estimator class

**v1.5**
    For methods/functions working on data matrices:
    the default input is a **n-by-p** 2D array (n samples from p discrete
    variables, with different samples on different **rows**).
    Since release 1.3, the default was a transposed (**p-by-n**) data matrix.
    The behavior of functions taking frequency counts as input
    (e.g. the *entropy* function) is unchanged.
**v1.4**
    Added the *kullback_leibler_divergence* function.
**v1.1**
    Added:

    * *from_data*
    * *mutual_information*
    * *conditional_information*
    * *interaction_information*
    * *coinformation*
**v1.0**
    Drop support for Python < 3.4.
**v0.9**
    Added the `jensen_shannnon_divergence` function.

References
==========

Some refs::

  @article{wolpert1995estimating,
    title={Estimating functions of probability distributions from a finite set of samples},
    author={Wolpert, David H and Wolf, David R},
    journal={Physical Review E},
    volume={52},
    number={6},
    pages={6841},
    year={1995},
    publisher={APS}
  }

  @inproceedings{nemenman2002entropy,
    title={Entropy and inference, revisited},
    author={Nemenman, Ilya and Shafee, Fariel and Bialek, William},
    booktitle={Advances in neural information processing systems},
    pages={471--478},
    year={2002}
  }

  @article{paninski2003estimation,
    title={Estimation of entropy and mutual information},
    author={Paninski, Liam},
    journal={Neural computation},
    volume={15},
    number={6},
    pages={1191--1253},
    year={2003},
    publisher={MIT Press}
  }

  @article{nemenman2004entropy,
    title={Entropy and information in neural spike trains: Progress on the sampling problem},
    author={Nemenman, Ilya and Bialek, William and van Steveninck, Rob de Ruyter},
    journal={Physical Review E},
    volume={69},
    number={5},
    pages={056111},
    year={2004},
    publisher={APS}
  }

  @article{nemenman2011coincidences,
    title={Coincidences and estimation of entropies of random variables with large cardinalities},
    author={Nemenman, Ilya},
    journal={Entropy},
    volume={13},
    number={12},
    pages={2013--2023},
    year={2011},
    publisher={Molecular Diversity Preservation International}
  }

  @article{archer2013bayesian,
    title={Bayesian and quasi-Bayesian estimators for mutual information from discrete data},
    author={Archer, Evan and Park, Il Memming and Pillow, Jonathan W},
    journal={Entropy},
    volume={15},
    number={5},
    pages={1738--1755},
    year={2013},
    publisher={Multidisciplinary Digital Publishing Institute}
  }

  @article{archer2014bayesian,
    title={Bayesian entropy estimation for countable discrete distributions},
    author={Archer, Evan and Park, Il Memming and Pillow, Jonathan W},
    journal={The Journal of Machine Learning Research},
    volume={15},
    number={1},
    pages={2833--2868},
    year={2014},
    publisher={JMLR. org}
  }


and interesting links:

- `Sebastian Nowozin on Bayesian estimators <http://www.nowozin.net/sebastian/blog/estimating-discrete-entropy-part-3.html>`_

- `Il Memming Park on discrete entropy estimators <https://memming.wordpress.com/2014/02/09/a-guide-to-discrete-entropy-estimators/>`_

Contributing
============

**ndd** is an OPEN Source Project so please help out by `reporting bugs <https://github.com/simomarsili/ndd>`_ or forking and opening pull requests when possible.

License
=======

Copyright (c) 2016-2019, Simone Marsili.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
