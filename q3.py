#!/bin/python2.7

import numpy as np

A = [[np.sqrt(1.0 - pow(-0.7 + 0.001 * (i - 1.0),2) - pow(-0.7 + 0.001 * (i - 1.0),2)) for i in range(1,1402)] for j in range(1,1402)]
A = np.mat(A)

