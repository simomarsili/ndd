# -*- coding: utf-8 -*-
from __future__ import absolute_import,division,print_function,unicode_literals
from builtins import *
import unittest
import ndd
import numpy as np
 
class TestNdd(unittest.TestCase):
 
    def test_1_100_100(self):
        a=1.0; ns=100; nd=100
        result = np.float64(4.322717746281727)
        np.random.seed(123)
        pp=np.random.dirichlet([a]*ns)
        data = np.random.multinomial(nd,pp)
        self.assertEqual(ndd.entropy(data), result)

    def test_01_100_100(self):
        a=0.1; ns=100; nd=100
        result = np.float64(2.569415088680637)
        np.random.seed(123)
        pp=np.random.dirichlet([a]*ns)
        data = np.random.multinomial(nd,pp)
        self.assertEqual(ndd.entropy(data), result)

    def test_001_100_100(self):
        a=0.01; ns=100; nd=100
        result = np.float64(0.21622160227928622)
        np.random.seed(123)
        pp=np.random.dirichlet([a]*ns)
        data = np.random.multinomial(nd,pp)
        self.assertEqual(ndd.entropy(data), result)

    def test_histogram(self):
        np.random.seed(123)
        data = np.random.randint(1,11,1000)
        _, h = np.unique(data, return_counts=True)
        self.assertEqual(ndd.histogram(data), list(h))

if __name__ == '__main__':
    unittest.main()

