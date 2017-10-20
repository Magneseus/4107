#!/bin/python2.7

import numpy as np

matrix = np.mat([[1,2,3], [2,3,4], [4,5,6], [1,1,1]])

V,S,U = np.linalg.svd(matrix, full_matrices=True)

print("U: ")
print(U)

print("\nS: ")
print(S)

print("\nV: ")
print(V)