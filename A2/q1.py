import os.path
import numpy as np
import scipy as sp
from sklearn.datasets import fetch_mldata

# Get the MNIST data, if not already there
DATA_DIR = "./"
mnist = fetch_mldata('MNIST original', data_home=DATA_DIR)

# Sigmoid activation function
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))