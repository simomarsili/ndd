# -*- coding: utf-8 -*-
# Copyright (C) 2016,2017 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""Base classes module."""


class BaseEstimator(object):
    # TODO: mimic sklearn BaseEstimator
    def __init__(self):
        self.estimate = None
        self.std = None

    def _check_input(self):
        # check input data
        raise NotImplementedError

    def fit(self):
        # set self.estimate, self.std
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Return estimate from input data. Delegate to fit."""
        self.fit(*args, **kwargs)
        return self.estimate
