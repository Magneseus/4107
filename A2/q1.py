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

np.seterr(over='raise')


# Sigmoid activation function
def sigmoid(x):
	x = np.clip(x, -500, 500)
	
	'''
	ret = 0

	try:
		ret = 1.0 / (1.0 + np.exp(-x))
	except:
		print(x)
	
	return ret
	'''
	return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
	return x * (1.0 - x)

# Softmax function
def softmax(sums):
	returnMat = np.zeros(shape=[len(sums), 1], dtype=np.float64)
	
	sums -= np.argmax(sums)

	# e^x / Sigma e^x
	returnMat = np.exp(sums) / np.sum(np.exp(sums), keepdims=True)

	return returnMat


class NeuralNet:
	def __init__(self, input_dim, num_hidden_layer, hidden_dim, output_dim):
		self.num_in  = input_dim
		self.num_hl  = num_hidden_layer
		self.num_hi  = hidden_dim
		self.num_out = output_dim

		self.rand = random.Random(None)

		# Define the matrices for the neuron outputs
		self.input_nodes  = np.zeros(shape=[1, self.num_in], dtype=np.float64)
		self.hidden_nodes = [np.zeros(shape=[1, self.num_hi], dtype=np.float64) for i in range(self.num_hl)]
		self.output_nodes = np.zeros(shape=[1, self.num_out], dtype=np.float64)

		# Define the matrices for the weights
		self.ih_weights = np.zeros(shape=[self.num_in, self.num_hi], dtype=np.float64)
		self.hh_weights = [np.zeros(shape=[self.num_hi, self.num_hi], dtype=np.float64) for i in range(self.num_hl-1)]
		self.ho_weights = np.zeros(shape=[self.num_hi, self.num_out], dtype=np.float64)

		# Define the matrices for the biases
		self.h_biases = [np.zeros(shape=[1, self.num_hi], dtype=np.float64) for i in range(self.num_hl)]
		self.o_biases = np.zeros(shape=[1, self.num_out], dtype=np.float64)

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
				(self.h_biases[ind])[0, i] = 0.02 * self.rand.random() - 0.01

		for i in range(self.num_out):
			self.o_biases[0, i] = 0.02 * self.rand.random() - 0.01

	def saveToFile(self, fileName='arrays'):
		dimensionDict = {}
		dimensionDict['ni'] = self.num_in
		dimensionDict['nl'] = self.num_hl
		dimensionDict['nh'] = self.num_hi
		dimensionDict['no'] = self.num_out

		np.savez(fileName, inNodes=nn.input_nodes, outNodes=nn.output_nodes, hNodes=nn.hidden_nodes, ihWeights=nn.ih_weights, hhWeights=nn.hh_weights, hoWeights=nn.ho_weights, hBiases=nn.h_biases, oBiases=nn.o_biases, dimDict=dimensionDict)

	def loadFromFile(self, fileName='arrays'):
		loaded = np.load(fileName + '.npz')

		self.input_nodes = loaded['inNodes']
		self.output_nodes = loaded['outNodes']
		self.hidden_nodes = loaded['hNodes']

		self.ih_weights = loaded['ihWeights']
		self.hh_weights = loaded['hhWeights']
		self.ho_weights = loaded['hoWeights']

		self.h_biases = loaded['hBiases']
		self.o_biases = loaded['oBiases']

		dimensionDict = loaded['dimDict'].item()
		self.num_in  = dimensionDict['ni']
		self.num_hl  = dimensionDict['nl']
		self.num_hi  = dimensionDict['nh']
		self.num_out = dimensionDict['no']



	def meanSquaredError(self, data, target):
		sumSquaredError = 0.0

		targMat = [np.zeros(shape=[1, self.num_out], dtype=np.float64) for i in range(len(target))]
		for i in range(len(target)):
			for j in range(self.num_out):
				(targMat[i])[0, j] = 1.0 if target[i] == j else 0.0

		for i in range(len(data)):
			guesses = self.feed_forward(data[i])
		
			for j in range(self.num_out):
				err = targMat[i][0, j] - guesses[0, j]
				sumSquaredError += err * err
			
		return sumSquaredError / len(data)

	def summedError(self, data, target):
		sum = 0.0
		for i in range(len(data)):
			guesses = self.feed_forward(data[i])

			guess = np.argmax(guesses)
			if (guess != target[i]):
				sum += 1.0

		return sum / len(data)

	def kfold(self, data, target, k):
		indList = range(len(data))
		self.rand.shuffle(indList)

		data2 = [data[i] for i in indList]
		target2 = [target[i] for i in indList]

		len_of_fold = int(np.floor(len(data) / float(k)))
		folds = [len_of_fold * i for i in range(k)]
		folds.append(len(data))

		errs = []

		for i in range(k):
			self.run(data2[0:folds[i]] + data2[folds[i+1]:len(data)], target2[0:folds[i]] + target2[folds[i+1]:len(data)], 1)

			errs.append(self.summedError(data2[folds[i]:folds[i+1]], target2[folds[i]:folds[i+1]]))

		return errs

	def feed_forward(self, input_vals):
		for i in range(self.num_in):
			self.input_nodes[0, i] = input_vals[i]

		self.hidden_nodes[0] = sigmoid(self.input_nodes.dot(self.ih_weights) + self.h_biases[0])

		for i in range(1, self.num_hl):
			self.hidden_nodes[i] = sigmoid(self.hidden_nodes[i-1].dot(self.hh_weights[i-1]) + self.h_biases[i])

		self.output_nodes = softmax(self.hidden_nodes[self.num_hl-1].dot(self.ho_weights) + self.o_biases)

		# return a copy of the results
		resultCopy = np.zeros(shape=[1, self.num_out], dtype=np.float64)
		for i in range(self.num_out):
			resultCopy[0, i] = self.output_nodes[0, i]

		return resultCopy

	def run(self, dataList=mnist.data, targetList=mnist.target, maxRuns=10, eps_learn=0.01, lam_decay=0.01):
		# Correction Matrices
		outputCorrections = np.zeros(shape=[1, self.num_out], dtype=np.float64)
		hiddenCorrections = [np.zeros(shape=[1, self.num_hi], dtype=np.float64) for i in range(self.num_hl)]

		# Data matrices and indices
		targMat = [np.zeros(shape=[self.num_out], dtype=np.float64) for i in range(len(targetList))]
		for i in range(len(targetList)):
			for j in range(self.num_out):
				(targMat[i])[j] = 1.0 if targetList[i] == j else 0.0

		# Loop until maxRuns is hit
		runs = 0
		while runs < maxRuns:
			n = 0

			# Go through all training items
			for ind in range(len(dataList)):
				if (n%10000 == 0):
					print("doing training num: %d  (%s)" % (n, datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')))
				n += 1

				#### Feed forward ####
				self.feed_forward(dataList[ind])

				#### Backpropagation ####

				### Correction calculations
				# Output corrections (softmax)
				outputCorrections = ( (1 - self.output_nodes) * self.output_nodes) * (targMat[ind] - self.output_nodes)

				# Hidden->Output corrections
				hiddenCorrections[self.num_hl-1] = outputCorrections.dot(self.ho_weights.T) * ((1 - self.hidden_nodes[self.num_hl-1]) * self.hidden_nodes[self.num_hl-1])

				# Hidden Corrections
				for hInd in reversed(range(self.num_hl-1)):
					hiddenCorrections[hInd] = hiddenCorrections[hInd+1].dot(self.hh_weights[hInd].T) * ((1 - self.hidden_nodes[hInd]) * self.hidden_nodes[hInd])

				### Regularization
				#doWeights = outputCorrections - (lam_decay * self.ho_weights)
				#dhWeights = [hiddenCorrections[i] - (lam_decay * self.hh_weights[i-1]) for i in range(1,self.num_hl)]
				#diWeights = hiddenCorrections[0] - (lam_decay * self.ih_weights)

				### Weight Updates
				# Output Weights
				self.ho_weights += eps_learn * np.dot(self.hidden_nodes[self.num_hl-1].T, outputCorrections)

				# Output Bias
				self.o_biases += eps_learn * np.sum(outputCorrections) * 1.0 # Or should this be -1?

				# Hidden weights
				for hInd in range(self.num_hl-1):
					self.hh_weights[hInd] += eps_learn * self.hidden_nodes[hInd].T.dot(hiddenCorrections[hInd+1])

				# Hidden bias
				for hInd in range(self.num_hl):
					self.h_biases[hInd] += eps_learn * np.sum(hiddenCorrections[hInd]) * 1.0 # Or should this be -1?

				# Input weights
				self.ih_weights += eps_learn * self.input_nodes.T.dot(hiddenCorrections[0])

			runs += 1


			#if (runs % 10 == 0):
			#print("finished run %d:   err: %.4f" % (runs, self.summedError(dataList, targetList)))
			#else:
			print('finished run %d' %(runs))


nn = NeuralNet(784, 3, 15, 10)
nn.loadFromFile()
#nn.run(mnist.data, mnist.target, 50, 0.01, 0.01)
nn.kfold(mnist.data, mnist.target, 5)