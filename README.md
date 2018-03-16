# ndd - Bayesian entropy estimation from discrete data

The **ndd** module provides a simple Python interface to an efficient 
implementation of the Nemenman-Schafee-Bialek (NSB) algorithm, 
a parameter-free, Bayesian entropy estimator for discrete data.

## Basic usage 

The `ndd.entropy()` function takes as input a vector of frequency counts 
(the observed frequencies for a set of classes or states) 
and returns an entropy estimate (in nats): 

```python
>>> counts
[7, 3, 5, 8, 9, 1, 3, 3, 1, 0, 2, 5, 2, 11, 4, 23, 5, 0, 8, 0]
>>> import ndd
>>> entropy_estimate = ndd.entropy(counts)
>>> entropy_estimate
2.623634344902917
```

Optionally, the uncertainty in the entropy estimate can be quantified 
by computing an approximation for the posterior standard deviation:

```python
>>> entropy_estimate, std = ndd.entropy(counts, return_std=True)
>>> std
0.048675500725595504
```

### Where to get it
Install using pip:

```bash
pip install -U ndd
```

or directly from sources in github for the latest version of the code:
```bash
pip install git+https://github.com/simomarsili/ndd.git
```

In order to compile **ndd**, you will need **numpy** (>= 1.9) and a
**Fortran compiler**  installed on your machine.
If you are using Debian or a Debian derivative such as Ubuntu,
you can install the gfortran compiler using the following command:

```bash
sudo apt-get install gfortran
```

### References

Some refs:

```
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
```

and interesting links:

- [Sebastian Nowozin on Bayesian estimators](http://www.nowozin.net/sebastian/blog/estimating-discrete-entropy-part-3.html)

- [Il Memming Park on discrete entropy estimators](https://memming.wordpress.com/2014/02/09/a-guide-to-discrete-entropy-estimators/)

### Contributing

**ndd** is an OPEN Source Project so please help out by [reporting bugs](https://github.com/simomarsili/ndd) or forking and opening pull requests when possible.

### License

Copyright (c) 2016,2017, Simone Marsili.  
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

