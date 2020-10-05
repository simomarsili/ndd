# Change log

## Unreleased

## [1.10] - 2020-10-05
### Changed
- `entropy()` function: renamed `nk` to `counts`
- removed the `zk` argument
- the signature of the *entropy* function is:
  ```
  entropy(counts, k=None, estimator='NSB', return_std=False)
  ```

## [1.9.2] - 2020-09-21
### Added
- the `entropy` function handles mappings and generators as input counts.

## [1.9.1] - 2020-07-07
### Fixed
- fix Counts.fit()

## [1.9] - 2020-06-25
### Changed
- if argument `k` is omitted, the `entropy` function will guess a reasonable
  alphabet size and select the best estimator for the sampling regime.

## [1.8.4] - 2020-06-24
### Changed
- the `entropy` function takes frequency counts (multiplicities) as input via
  the `zk` optional argument
- renamed `pk` to `nk`
- the signature of the *entropy* function is:
  ```
  entropy(nk, k=None, zk=None, estimator='NSB', return_std=False)
  ```

## [1.8.3] - 2020-06-16
### Fixed
- integration for large cardinalities

## [1.8] - 2020-06-10
### Added
- full Bayesian error estimate (from direct computation of the posterior
  variance of the entropy)

## [1.7] - 2020-06-09
### Changed
- rename version.json to package.json
- check_pk: elements of input counts array are converted to int32 with no check
- entropy() function: NSB estimator needs cardinality `k` as input
- entropy() function: non-flat arrays are valid input counts
### Fixed
- robust integration range

## [1.6.4] - 2020-05-07
### Fixed
- fix coinformation function for combinations of features
- fix from_data for scalar inputs

## [1.6.1] - 2019-08-11
### Changed
For the NSB integration:
- Find the saddle point maximizing p(alpha | data)
- Set the integration range at 4 standard deviations around the saddle point
- If the standard deviation is too small, return the entropy posterior mean at
  the saddle point

## [1.6] - 2019-08-09
### Added
- `MillerMadow` estimator class
- `AsymptoticNSB` estimator class
- `Grassberger` estimator class
### Changed
The signature of the *entropy* function has been changed to allow
arbitrary entropy estimators. The new signature is
```
entropy(pk, k=None, estimator='NSB', return_std=False)
```
Check `ndd.entropy_estimators` for the available estimators.

## [1.5] - 2019-08-02
### Changed
For methods/functions working on data matrices:
the default input is a **n-by-p** 2D array (n samples from p discrete
variables, with different samples on different **rows**).
Since release 1.3, the default was a transposed (**p-by-n**) data matrix.
The behavior of functions taking frequency counts as input
(e.g. the `ndd.entropy` function) is unchanged.
### Added
- builds on Windows (with MinGW-W64)
- builds on MacOS (thanks to https://github.com/ccattuto)

## [1.4] - 2019-05-20
### Added
- `ndd.kullback_leibler_divergence`

## [1.3.2] - 2019-05-16
### Changed
- `r` (length of combinations) defaults to None

## [1.3] - 2019-05-08
### Changed
- input data arrays must be p-by-n 2D ndarrays containing
  n samples from p discrete variables. This affects all methods/functions
  working directly on data:
  - histogram
  - from_data
  - interaction_information
  - coinformation
  - mutual_information
  - conditional_entropy

## [1.2] - 2019-04-01
### Changed
- fixed conditional_entropy function
- histogram: `axis` should be None is data matrix is tarnsposed.

## [1.1] - 2019-03-26
### Added
- `ndd.from_data`
- `ndd.mutual_information`
- `ndd.conditional_information`
- `ndd.interaction_information`
- `ndd.coinformation`

## [1.0] - 2019-03-19
### Changed
- Python3 only
- fixed NumericError for valid entropy estimation

## [0.9] - 2019-01-15
### Added
- `jensen_shannnon_divergence()` function.
- `estimators` module.
- Entropy, KLDivergence and JSDivergence estimators.
### Changed
- entropy() signature is now:
  `entropy(pk, k=None, alpha=None, plugin=False, return_std=False)`

## [0.8] - 2019-01-08
### Added
- Entropy class.
- from_data() returns an estimate from a data samples

### Changed
- histogram(): remove unique elements of array as optional output
- histogram(): takes the `axis` and `r` optional args

## [0.7.1] - 2018-06-22
### Changed
- entropy(): removed the (broken) function for estimator selection

## [0.7] - 2018-03-22
### Changed
- entropy(): renamed argument 'dist' to 'plugin'

## [0.6] - 2018-03-22
### Changed
- entropy(): renamed argument 'a' to 'alpha'

## [0.5.5] - 2018-03-16
### Changed
- fix missing README.rst
- handle input counts arrays > 1D (always flatten to 1D)
- handle single-class case

## [0.5.3] - 2018-03-15
### Changed
- require numpy >= 1.9

## [0.5] - 2017-11-15
### Changed
- now a Python package
- binary extension are installed in the package dir

## [0.4.1] - 2017-11-14
### Added
- version.json

### Removed
- version.py

## [0.4] - 2017-10-03
### Changed
- numerical integration is stable over large ranges of conditions (large k and/or large n)
- raise an error if estimate is NaN
- statndard implementation for ML estimator

### Removed
- preliminary notebooks

## [0.3] - 2017-09-29
### Added
- Manifest.in

### Changed
- Changed `return_error=` to `return_std=` as key in the kwargs of ndd.entropy()
- Renamed `CHANGES.md` to `CHANGELOG.md`
- Check for NumPy version before setup
