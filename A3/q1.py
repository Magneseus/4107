import numpy as np
import scipy as sp
import tensorflow as tf
import random
from sklearn.datasets import fetch_mldata

rnd = random.Random()

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
def weight_training_zero(matrix):
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
def weight_training_neg(matrix):
	y = np.zeros((matrix.shape[1], matrix.shape[1]))
	for i in range(matrix.shape[0]):
		y += matrix[i][np.newaxis] * matrix[i][np.newaxis].T

	y -= np.identity(matrix.shape[1]) * matrix.shape[0]

	return y

x = weight_training_zero(x)
x2 = weight_training_neg(x2)
y = np.array([-1.,1.,1.,1.,1.,-1.])

def update(weights, vec, ind, neg=True):
	altNum = -1. if neg else 0.
	vector = np.copy(vec)

	sum = 0.
	for j in range(vector.size):
		if j != ind:
			sum += weights[ind][j] * vector[j]

	if sum >= 0.:
		vector[ind] = 1.
	else:
		vector[ind] = altNum

	return vector

def hopfield(weights, vec, cap=-1):
	iterations = 0
	vector = np.copy(vec)
	old_vector = np.zeros(vector.size)
	unchanged = True

	indices = range(vector.size)
	rnd.shuffle(indices)

	while (cap == -1 or iterations < cap):
		old_vector = vector

		# Pick a random neuron to update
		index = indices[0]
		indices = np.delete(indices, 0)

		if indices.size == 0:
			indices = range(vector.size)
			rnd.shuffle(indices)

			if unchanged:
				break

			unchanged = True

		# Update using that index
		vector = update(weights, vector, index)

		if not (np.array_equal(old_vector, vector)):
			unchanged = False

		iterations += 1

	return vector