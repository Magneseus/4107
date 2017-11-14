import numpy as np
import scipy as sp
import tensorflow as tf
import random
from sklearn.datasets import fetch_mldata

# Get the MNIST data, if not already there
DATA_DIR = "./"
mnist = fetch_mldata('MNIST original', data_home=DATA_DIR)

# Initiate the random generator
rnd = random.Random(3)

##### TEMP STUFF REMOVE LATER ######

x = np.zeros([5,6])

x[0] = np.array([0., 0., 1., 0., 1., 0.])
x[1] = np.array([1., 1., 1., 1., 0., 0.])
x[2] = np.array([1., 0., 1., 1., 1., 0.])
x[3] = np.array([0., 1., 0., 0., 0., 1.])
x[4] = np.array([0., 1., 1., 0., 0., 0.])

x2 = np.zeros([5,6])

x2[0] = np.array([-1., -1.,  1., -1.,  1., -1.])
x2[1] = np.array([ 1.,  1.,  1.,  1., -1., -1.])
x2[2] = np.array([ 1., -1.,  1.,  1.,  1., -1.])
x2[3] = np.array([-1.,  1., -1., -1., -1.,  1.])
x2[4] = np.array([-1.,  1.,  1., -1., -1., -1.])

##### END OF TEMP STUFF TO REMOVE LATER ######

def binarize(vals, thresh=1):
	return np.array([(-1. if val < thresh else 1.) for val in vals])

# Separate the mnist data into two arrays
mnist_train_1 = np.array([mnist.data[i] for i in range(60000) if mnist.target[i] == 1.0])
mnist_train_5 = np.array([mnist.data[i] for i in range(60000) if mnist.target[i] == 5.0])
mnist_test_1  = np.array([mnist.data[i] for i in range(60000, 70000) if mnist.target[i] == 1.0])
mnist_test_5  = np.array([mnist.data[i] for i in range(60000, 70000) if mnist.target[i] == 5.0])

# We can store 0.185 * 784 ~= 144 samples
# Store 72 from each set, chosen randomly
mnist_train_subset = np.array([(binarize(mnist_train_1[i//2]) if i % 2 == 0 else binarize(mnist_train_5[i//2])) for i in range(144)])
rnd.shuffle(mnist_train_subset)

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

	print("iterations" + str(iterations))
	return vector

w = weight_training_neg(mnist_train_subset)
out = hopfield(w, binarize(mnist_test_5[1]))

def print_out_img(out_vec):
	for i in range(28):
		s = ""
		for j in range(28):
			s += '0' if out_vec[(i*28)+j] == -1 else '1'
			s += " "
		print(s)

print_out_img(out)