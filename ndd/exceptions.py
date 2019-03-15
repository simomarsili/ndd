# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""Base EntropyEstimator class."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import (  # pylint: disable=redefined-builtin, unused-import
    bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next,
    oct, open, pow, round, super, filter, map, zip)
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


class PMFError(EstimatorInputError):
    """Invalid PMF."""


class CardinalityError(EstimatorInputError):
    """Invalid size of the sample space (cardinality)."""


class NumericError(NddError):
    """Numeric error during estimation."""


class HistogramError(NddError):
    """Error during evaluation of histogram."""


class AxisError(NddError):
    """invalid axis."""
