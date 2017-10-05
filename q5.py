#!/bin/python2.7

import numpy as np


# Functions

def nullspace(A, tolerance=1e-5):
    # SVD
    u, s, v = np.linalg.svd(A, full_matrices=True)

    # Number of columns to skip
    null_mask = (s >= tolerance).sum()
    
    # Take only remaining columns and transpose
    return v[null_mask:].T


A = np.mat([[3, 2, -1, 4], [1, 0, 2, 3], [-2, -2, 3, -1]])
A_null = nullspace(A)

print("A_Nullspace: \n{0}\n\n".format(A_null))

AT_null = nullspace(A.T)

print("AT_Nullspace: \n{0}\n".format(AT_null))