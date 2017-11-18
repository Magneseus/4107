import numpy as np
import random
from sklearn.datasets import fetch_mldata

# Get the MNIST data, if not already there
DATA_DIR = "./"
mnist = fetch_mldata('MNIST original', data_home=DATA_DIR)

# Initiate the random generator
rnd = random.Random(10)

# Small helper function to take image data and binarize it
def binarize(vals, thresh=1):
	return np.array([(-1. if val < thresh else 1.) for val in vals])

# Separate the mnist data into two arrays
mnist_train_1 = np.array([mnist.data[i] for i in range(60000) if mnist.target[i] == 1.0])
mnist_train_5 = np.array([mnist.data[i] for i in range(60000) if mnist.target[i] == 5.0])
mnist_test_1  = np.array([binarize(mnist.data[i]) for i in range(60000, 70000) if mnist.target[i] == 1.0])
mnist_test_5  = np.array([binarize(mnist.data[i]) for i in range(60000, 70000) if mnist.target[i] == 5.0])

# Number of images to feed to the neural network
num_samples = 10

# We can store maximum (0.185 * 784 ~= 144) samples
# Store 72 from each set, chosen randomly
mnist_train_subset = np.array([(binarize(mnist_train_1[rnd.randint(0,mnist_train_1.shape[0])]) if i % 2 == 1 else binarize(mnist_train_5[rnd.randint(0,mnist_train_5.shape[0])])) for i in range(num_samples)])

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

def storkey_training(matrix):
	old_weights = None

	for i in range(len(matrix)):
		old_weights = storkey_training_single(matrix[i], old_weights)

	return old_weights

def storkey_training_single(vec, old_weights=None):
    # If we have no previous weight vec, generate one
    if type(old_weights) == type(None):
        old_weights = np.zeros(len(vec))

    hebbian_term  = np.outer(vec,vec) - np.identity(len(vec))

    net_inputs    = old_weights.dot(vec)

    pre_synaptic  = np.outer(vec,net_inputs)
    post_synaptic = pre_synaptic.T

    new_weights = old_weights + (1./len(vec))*(hebbian_term - pre_synaptic - post_synaptic)

    return new_weights

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
		vector[ind] = -1.

	return vector

def hopfield(weights, vec, cap=-1):
	iterations = 0
	vector = np.copy(vec)
	old_vector = np.zeros(vector.size)
	unchanged = True

	#indices = range(vector.size)
	#rnd.shuffle(indices)
	sampled = 0
	max_sampled = vector.size * 2

	while (cap == -1 or iterations < cap):
		old_vector = vector

		# Pick a random neuron to update
		index = int(rnd.random() * (vector.size-1))#indices[0]
		#indices = np.delete(indices, 0)

		if sampled == max_sampled:#indices.size == 0:
			#indices = range(vector.size)
			#rnd.shuffle(indices)

			if unchanged:
				break

			unchanged = True
			sampled = 0

		# Update using that index
		vector = update(weights, vector, index)

		if not (np.array_equal(old_vector, vector)):
			unchanged = False

		iterations += 1
		sampled += 1

	#print("iterations" + str(iterations))
	return vector

def get_smallest_distance(input_vector):
	min_dist = np.finfo(np.float32).max
	min_dist_num = -1

	# Check 1's
	for vec in mnist_test_1:
		dist = np.linalg.norm(input_vector - vec)

		if dist < min_dist:
			min_dist = dist
			min_dist_num = 1

	# Check 5's
	for vec in mnist_test_5:
		dist = np.linalg.norm(input_vector - vec)

		if dist < min_dist:
			min_dist = dist
			min_dist_num = 5

	return min_dist_num

# Train the network
w = storkey_training(mnist_train_subset)

# Test the network
num_correct = 0
max_correct = len(mnist_test_1) + len(mnist_test_5)

# Test 5's
def test_5(num_correct=num_correct, max_correct=max_correct):
	for i in range(len(mnist_test_5)):
		# Get the output of a vector
		out_vec = hopfield(w, mnist_test_5[i])

		# Check what it's classified as
		if get_smallest_distance(out_vec) == 5:
			num_correct += 1

		print('{0}/{1}  ({2})'.format(i+1, max_correct, num_correct))

# Test 1's
def test_1(num_correct=num_correct, max_correct=max_correct):
	for i in range(len(mnist_test_1)):
		# Get the output of a vector
		out_vec = hopfield(w, mnist_test_1[i])

		# Check what it's classified as
		if get_smallest_distance(out_vec) == 1:
			num_correct += 1

		print('{0}/{1}  ({2})'.format(i+1, max_correct, num_correct))

def percent(max):
	print('Percentage correct: {0}%'.format(num_correct/float(max)))


def print_out_img(out_vec):
	for i in range(28):
		s = ""
		for j in range(28):
			s += ' ' if out_vec[(i*28)+j] == -1 else '1'
			s += " "
		print(s)

#print_out_img(out)
#print_out_img(hopfield(w, binarize(mnist_test_1[1])))