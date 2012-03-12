'''
This file is part of the Data Analysis Recipes project.
Copyright 2011 David W. Hogg (NYU)
'''

#!/usr/bin/env python
import numpy as np
import unittest

import hogg_rv

class testCV(unittest.TestCase):
    def setUp(self):
        self.data, self.info = hogg_rv.read_data('HD104067.dat')
        self.model = 'sinusoid+jitter'
        self.pars = np.array([1.51226917e+01, 8.87236709e-03, 8.84415337e-03, 1.12435113e-01])
        self.pars = np.append(self.pars, [0.01])

    def test_seven(self):
        ll = hogg_rv.loo_lnlikelihood(self.pars, self.data, self.model, self.info, 7)
        self.assertAlmostEqual(ll, 4.7325645019)

if __name__ == '__main__':
    unittest.main()
