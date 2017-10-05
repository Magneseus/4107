#!/bin/python2.7

import numpy as np

# SVD of A 1401
A = [[np.sqrt(1.0 - pow(-0.7 + 0.001 * (i - 1.0),2) - pow(-0.7 + 0.001 * (i - 1.0),2)) for i in range(1,1402)] for j in range(1,1402)]
A = np.mat(A)

U,S,V = np.linalg.svd(A, full_matrices=True)


# Low rank approx (2) of A
A2 = np.zeros((len(U), len(V)))
for i in range(2):
	A2 += S[i] * np.outer(U.T[i], V[i])

print(A2)
print("Rank(A2): ",  np.ndim(A2))