# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
# pylint: disable=line-too-long
"""
# ndd - Bayesian entropy estimation from discrete data

The **ndd** module provides a simple Python interface to an efficient
implementation of the Nemenman-Schafee-Bialek (NSB) algorithm,
a parameter-free, Bayesian entropy estimator for discrete data.

## Basic usage

The `ndd.entropy()` function takes as input a vector of frequency counts
(the observed frequencies for a set of classes or states)
and returns an entropy estimate (in nats):

```
>>> counts = [7, 3, 5, 8, 9, 1, 3, 3, 1, 0, 2, 5, 2, 11, 4, 23, 5, 0, 8, 0]
>>> import ndd
>>> entropy_estimate = ndd.entropy(counts)
>>> entropy_estimate
2.6017414378907606

```

Optionally, the uncertainty in the entropy estimate can be quantified
by computing an approximation for the posterior standard deviation:

```
>>> entropy_estimate, std = ndd.entropy(counts, return_std=True)
>>> std
0.04090929117340808

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
import pkg_resources

from ndd import package_setup
from ndd.counts import CountsDistribution
from ndd.divergence import DivergenceEstimator
from ndd.divergence import estimators as divergence_estimators
from ndd.estimators import EntropyEstimator
from ndd.estimators import estimators as entropy_estimators
from ndd.nsb import entropy  # pylint: disable=unused-import
from ndd.nsb import (coinformation, conditional_entropy, from_data, histogram,
                     interaction_information, jensen_shannon_divergence,
                     kullback_leibler_divergence, mutual_information)

package_name = package_setup.package_name
package_path = package_setup.package_path
__version__ = pkg_resources.require(package_name)[0].version
__all__ = [
    'coinformation', 'conditional_entropy', 'entropy', 'from_data',
    'histogram', 'interaction_information', 'jensen_shannon_divergence',
    'kullback_leibler_divergence', 'mutual_information', 'CountsDistribution'
]
