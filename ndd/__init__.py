# -*- coding: utf-8 -*-
# Copyright (C) 2017, Simone Marsili
# All rights reserved.
# License: BSD 3 clause
# Author: Simone Marsili (simomarsili@gmail.com)
"""
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

"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)
import os
import json
from ndd.nsb import (entropy, histogram)

path_to_version = os.path.join(os.path.dirname(__file__), 'version.json')
with open(path_to_version, 'r') as f:
    version_data = json.load(f)
    try:
        __version__ = version_data['version']
    except KeyError:
        # no version number in version.json
        raise KeyError("check version file: no version number")

__copyright__ = "Copyright (C) 2016,2017 Simone Marsili"
__license__ = "BSD 3 clause"
__author__ = "Simone Marsili (simo.marsili@gmail.com)"
__all__ = ["entropy",
           "histogram"]
