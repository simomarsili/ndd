# -*- coding: utf-8 -*-
# Author: Simone Marsili <simomarsili@gmail.com>
# License: BSD 3 clause
"""Base EntropyEstimator class."""
import logging


logger = logging.getLogger(__name__)


class NddError(Exception):
    """Base class for ndd exceptions."""


class EstimatorParameterError(NddError):
    """Getting/setting invalid estimator parameters."""


class AlphaError(EstimatorParameterError):
    """Invalid size of the sample space (cardinality)."""


class EstimatorInputError(NddError):
    """Getting/setting invalid estimator parameters."""


class CountsError(EstimatorInputError):
    """Invalid counts array."""


class PmfError(EstimatorInputError):
    """Invalid PMF."""


class CardinalityError(EstimatorInputError):
    """Invalid size of the sample space (cardinality)."""


class NumericError(NddError):
    """Numeric error during estimation."""


class HistogramError(NddError):
    """Error during evaluation of histogram."""


class AxisError(NddError):
    """invalid axis."""
