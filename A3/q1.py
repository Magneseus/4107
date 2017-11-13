import numpy as np
import scipy as sp
import tensorflow as tf
from sklearn.datasets import fetch_mldata

x = np.zeros([5,6])

x[0] = np.array([0., 0., 1., 0., 1., 0.])
x[1] = np.array([1., 1., 1., 1., 0., 0.])
x[2] = np.array([1., 0., 1., 1., 1., 0.])
x[3] = np.array([0., 1., 0., 0., 0., 1.])
x[4] = np.array([0., 1., 1., 0., 0., 0.])

x2 = np.zeros([5,6])

x2[0] = np.array([-1., -1., 1., -1., 1., -1.])
x2[1] = np.array([1., 1., 1., 1., -1., -1.])
x2[2] = np.array([1., -1., 1., 1., 1., -1.])
x2[3] = np.array([-1., 1., -1., -1., -1., 1.])
x2[4] = np.array([-1., 1., 1., -1., -1., -1.])


# Creates the initial weights for a set of training examples
# Needs to be in a binary 1,0 format (see matrix: x)
def weight_training_1zero(matrix):
	weights = np.zeros((matrix.shape[1], matrix.shape[1]))

	for i in range(matrix.shape[1]-1):
		for j in range(i+1, matrix.shape[1]):
			sum = 0

			for k in range(0, matrix.shape[0]):
				if matrix[k][i] == matrix[k][j]:
					sum = sum + 1
				else:
					sum = sum - 1

			weights[i][j] = sum

	return weights + weights.T

# Creates the initial weights for a set of training examples
# Needs to be in a binary 1,-1 format (see matrix: x2)
def weight_training_1neg(matrix):
	y = np.zeros((matrix.shape[1], matrix.shape[1]))
	for i in range(matrix.shape[0]):
		y += matrix[i][np.newaxis] * matrix[i][np.newaxis].T

	y -= np.identity(matrix.shape[1]) * matrix.shape[0]

	return y
