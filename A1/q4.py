#!/bin/python2.7

import numpy as np

A = np.mat([[1,2,3], [2,3,4], [4,5,6], [1,1,1]])

B = np.mat([1,1,1,1])
B = B.T

epsil = [0.01, 0.05, 0.1, 0.15, 0.2]
delta = 0.01


Xvals = []
iterations = []

for e in range(len(epsil)):
	X = 1
	it = 0
	
	while np.linalg.norm((A.T * A * X) - (A.T * B)) > delta:
		X = X - epsil[e] * ((A.T * A * X) - (A.T * B))
		it += 1

	X2 = []
	for i in range(len(X[:,:1].tolist())):
		X2.append(X[:,:1].tolist()[i][0])

	Xvals.append(X2)
	iterations.append(it)

print("Ignore the errors! :D\n\n")

for i in range(len(epsil)):
	print("e: {0}\nx: {1}\niter: {2}\n".format(epsil[i], Xvals[i], iterations[i]))