# -*- coding: utf-8 -*-
from __future__ import print_function
import unittest
import ndd
import numpy as np
 
class TestNdd(unittest.TestCase):
 
    def test_1_100_100(self):
        # generate your own data for testing: 
        # import numpy as np; a=1.0; ns=100; nd=100; al = [a]*ns; pp=np.random.dirichlet(al); np.random.multinomial(nd,pp)
        data = np.array([1, 1, 0, 0, 4, 2, 0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 4, 3,
                         0, 0, 1, 1, 0, 3, 3, 3, 0, 1, 5, 0, 0, 0, 0, 2, 0, 0, 3, 1, 0, 0, 3,
                         0, 1, 2, 0, 1, 1, 2, 2, 1, 2, 0, 0, 3, 0, 2, 1, 6, 2, 0, 6, 0, 1, 0,
                         0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 0, 1, 1, 5, 1, 0, 0, 0,
                         0, 1, 0, 1, 2, 1, 0, 0])
        result = np.float64(4.194084935806322)
        self.assertEqual(ndd.entropy(data), result)

    def test_01_100_100(self):
        # generate your own data for testing: 
        # import numpy as np; a=0.1; ns=100; nd=100; al = [a]*ns; pp=np.random.dirichlet(al); np.random.multinomial(nd,pp)
        data = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  2,  0,
                          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                          3,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0,
                          0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  2,  0, 18,  0,  0,  0,
                          0,  0,  3,  0, 43,  0,  0,  0,  0,  0,  0,  8,  0,  2,  0,  0,  0,
                          0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  3,  1,  0,  0])
        result = np.float64(2.1324484952916856)
        self.assertEqual(ndd.entropy(data), result)

    def test_001_100_100(self):
        # generate your own data for testing: 
        # import numpy as np; a=0.01; ns=100; nd=100; al = [a]*ns; pp=np.random.dirichlet(al); np.random.multinomial(nd,pp)
        data = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 88,  0,  0,  0,  0,  0,
                          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  0,  0,  0,  0,
                          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,
                          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
        result = np.float64(0.45816599887523507)
        self.assertEqual(ndd.entropy(data), result)

if __name__ == '__main__':
    unittest.main()

