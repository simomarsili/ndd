# Change log

## [Unreleased]
### Added
- Entropy class. 
- from_data() returns an estimate from a data samples

### Changed
- histogram(): remove unique elements of array as optional output

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
