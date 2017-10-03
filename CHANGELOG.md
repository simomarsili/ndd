# Change log

## [Unreleased]

## [0.4] - 2017-10-03
### [Changed]
- numerical integration is stable over large ranges of conditions (large k and/or large n)
- raise an error if estimate is NaN
- statndard implementation for ML estimator

### [Removed]
- preliminary notebooks

## [0.3] - 2017-09-29
### [Added]
- Manifest.in

### [Changed]
- Changed `return_error=` to `return_std=` as key in the kwargs of ndd.entropy()
- Renamed `CHANGES.md` to `CHANGELOG.md`
- Check for NumPy version before setup
