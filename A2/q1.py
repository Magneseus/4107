import numpy as np
import scipy as sp
import random
import time
import datetime
from sklearn.datasets import fetch_mldata

# Get the MNIST data, if not already there
DATA_DIR = "./"
mnist = fetch_mldata('MNIST original', data_home=DATA_DIR)

# Hyperparameters
input_dim = 784 # (28x28 image)
output_dim = 10 # (0-9)

eps = 0.01 # Learning rate
lam = 0.01 # Decay lambda


# Sigmoid activation function
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

# Softmax function
def softmax(sums):
	returnMat = np.zeros(shape=[len(sums)], dtype=np.float32)
	
	# Calculate e^z
	for i in range(len(sums)):
		returnMat[i] = np.exp(sums[i])

	# Sum
	esum = np.sum(returnMat)

	for i in range(len(sums)):
		returnMat[i] = returnMat[i] / esum

	return returnMat


class NeuralNet:
	def __init__(self, input_dim, num_hidden_layer, hidden_dim, output_dim):
		self.num_in  = input_dim
		self.num_hl  = num_hidden_layer
		self.num_hi  = hidden_dim
		self.num_out = output_dim

		self.rand = random.Random(None)

		# Define the matrices for the neuron outputs
		self.input_nodes  = np.zeros(shape=[self.num_in], dtype=np.float32)
		self.hidden_nodes = [np.zeros(shape=[self.num_hi], dtype=np.float32) for i in range(self.num_hl)]
		self.output_nodes = np.zeros(shape=[self.num_out], dtype=np.float32)

		# Define the matrices for the weights
		self.ih_weights = np.zeros(shape=[self.num_in, self.num_hi], dtype=np.float32)
		self.hh_weights = [np.zeros(shape=[self.num_hi, self.num_hi], dtype=np.float32) for i in range(self.num_hl-1)]
		self.ho_weights = np.zeros(shape=[self.num_hi, self.num_out], dtype=np.float32)

		# Define the matrices for the biases
		self.h_biases = [np.zeros(shape=[self.num_hi], dtype=np.float32) for i in range(self.num_hl)]
		self.o_biases = np.zeros(shape=[self.num_out], dtype=np.float32)

		# Initialize weights to random values
		for i in range(self.num_in):
			for j in range(self.num_hi):
				self.ih_weights[i,j] = 0.02 * self.rand.random() - 0.01

		for ind in range(self.num_hl-1):
			for i in range(self.num_hi):
				for j in range(self.num_hi):
					(self.hh_weights[ind])[i,j] = 0.02 * self.rand.random() - 0.01

		for i in range(self.num_hi):
			for j in range(self.num_out):
				self.ho_weights[i,j] = 0.02 * self.rand.random() - 0.01

		# Initialize biases to random values
		for ind in range(self.num_hl):
			for i in range(self.num_hi):
				(self.h_biases[ind])[i] = 0.02 * self.rand.random() - 0.01

		for i in range(self.num_out):
			self.o_biases[i] = 0.02 * self.rand.random() - 0.01

	def feed_forward(self, input_vals):
		# tmp matrices to store values
		hidden_sums = np.zeros(shape=[self.num_hi], dtype=np.float32)
		output_sums  = np.zeros(shape=[self.num_out], dtype=np.float32)

		# Copy input values
		for i in range(self.num_in):
			self.input_nodes[i] = input_vals[i]

		### First hidden layer
		# Weights
		for i in range(self.num_hi):
			for j in range(self.num_in):
				hidden_sums[i] = self.input_nodes[j] * self.ih_weights[j,i]

		# Biases
		for i in range(self.num_hi):
			hidden_sums[i] += (self.h_biases[0])[i]

		# Activation
		for i in range(self.num_hi):
			(self.hidden_nodes[0])[i] = sigmoid(hidden_sums[i])


		### All remaining hidden layers
		for ind in range(self.num_hl-1):
			# Weights
			for i in range(self.num_hi):
				for j in range(self.num_hi):
					hidden_sums[i] = (self.hidden_nodes[ind])[j] * (self.hh_weights[ind])[j,i]

			# Biases
			for i in range(self.num_hi):
				hidden_sums[i] += (self.h_biases[ind+1])[i]

			# Activation
			for i in range(self.num_hi):
				(self.hidden_nodes[ind+1])[i] = sigmoid(hidden_sums[i])

		### Output Layer
		# Weights
		for i in range(self.num_out):
			for j in range(self.num_hi):
				output_sums[i] = (self.hidden_nodes[self.num_hl-1])[j] * self.ho_weights[j,i]

		# Biases
		for i in range(self.num_out):
			output_sums[i] += self.o_biases[i]

		# Activation
		softMaxOut = softmax(output_sums)
		for i in range(self.num_out):
			self.output_nodes[i] = softMaxOut[i]


		# return a copy of the results
		resultCopy = np.zeros(shape=self.num_out, dtype=np.float32)
		for i in range(self.num_out):
			resultCopy[i] = self.output_nodes[i]

		return resultCopy

	def run(self, dataList, targetList, maxRuns, eps_learn, lam_decay):
		# Correction Matrices
		outputCorrections = np.zeros(shape=[self.num_out], dtype=np.float32)
		hiddenCorrections = [np.zeros(shape=[self.num_hi], dtype=np.float32) for i in range(self.num_hl)]

		# Data matrices and indices
		targMat = [np.zeros(shape=[self.num_out], dtype=np.float32) for i in range(len(targetList))]
		for i in range(len(targetList)):
			for j in range(self.num_out):
				(targMat[i])[j] = 1.0 if targetList[i] == j else 0.0

		indList = range(len(dataList))

		# Loop until maxRuns is hit
		runs = 0
		while runs < maxRuns:
			# TODO: Change to a k-fold cross
			self.rand.shuffle(indList)

			n = 0

			# Go through all training items
			for ind2 in range(len(dataList)):
				ind = indList[ind2]

				if (n%1000 == 0):
					print("doing training num: %d  (%s)" % (n, datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')))
				n += 1

				#### Feed forward ####
				self.feed_forward(dataList[ind])

				#### Backpropagation ####

				### Correction calculations
				# Output corrections (softmax)
				for i in range(self.num_out):
					outputCorrections[i] = ( (1 - self.output_nodes[i]) * self.output_nodes[i]) * ((targMat[ind])[i] - self.output_nodes[i])

				# Hidden corrections (sigmoid)
				for hInd in reversed(range(self.num_hl)):
					for i in range(self.num_hi):
						# Sum up corrections
						corSum = 0.0

						# If we're the last hidden layer, adjust values to the output layer
						last = hInd == self.num_hl-1
						count = self.num_out if last else self.num_hi
						for j in range(count):
							if not last:
								corSum += ((self.hh_weights[hInd])[i,j] * (hiddenCorrections[hInd+1])[j])
							else:
								corSum += (self.ho_weights[i,j] * outputCorrections[j])

						(hiddenCorrections[hInd])[i] = (1 - (self.hidden_nodes[hInd])[i]) * (self.hidden_nodes[hInd])[i] * corSum

				### Weight Updates
				# Output Weights
				for i in range(self.num_hi):
					for j in range(self.num_out):
						self.ho_weights[i,j] += eps_learn * outputCorrections[j] * (self.hidden_nodes[self.num_hl-1])[i]

				# Output Bias
				for i in range(self.num_out):
					self.o_biases[i] += eps_learn * outputCorrections[i] * 1.0 # Or should this be -1?

				# Hidden weights
				for hInd in range(self.num_hl-1):
					for i in range(self.num_hi):
						for j in range(self.num_hi):
							(self.hh_weights[hInd])[i,j] += eps_learn * (self.hidden_nodes[hInd])[i] * (hiddenCorrections[hInd+1])[j]

				# Hidden bias
				for hInd in range(self.num_hl-1):
					for i in range(self.num_hi):
						(self.h_biases[hInd]) += eps_learn * (hiddenCorrections[hInd])[i] * 1.0 # Or should this be -1?

				# Input weights
				for i in range(self.num_in):
					for j in range(self.num_hi):
						self.ih_weights[i,j] += eps_learn * self.input_nodes[i] * (hiddenCorrections[0])[j]

			runs += 1

			if (runs % 1 == 0):
				print("finished run: %d", runs)


nn = NeuralNet(784, 2, 10, 10)
nn.run(mnist.data, mnist.target, 10, 0.01, 0.01)