import numpy as np
import random
from sklearn.datasets import fetch_mldata

# Get the MNIST data, if not already there
DATA_DIR = "./"
mnist = fetch_mldata('MNIST original', data_home=DATA_DIR)

# Initiate the random generator
rnd = random.Random(3)

# Separate the mnist data into two arrays
mnist_train_1 = np.array([mnist.data[i] for i in range(60000) if mnist.target[i] == 1.0])
mnist_train_5 = np.array([mnist.data[i] for i in range(60000) if mnist.target[i] == 5.0])
mnist_test_1  = np.array([mnist.data[i] for i in range(60000, 70000) if mnist.target[i] == 1.0])
mnist_test_5  = np.array([mnist.data[i] for i in range(60000, 70000) if mnist.target[i] == 5.0])

x = np.concatenate((mnist_train_1, mnist_train_5))

def kmeans(data, k, lam=0.001, lam_dec=0.00005):
	# Pick some centroids
	centroids = np.array([np.copy(data[rnd.randint(0,data.shape[0]-1)])])
	for i in range(k-1):
		centroids = np.concatenate((centroids, np.array([np.copy(data[rnd.randint(0,data.shape[0]-1)])])))

	while (lam > 0.0):
		for i in range(data.shape[0]):
			closest_dist = np.finfo(np.float32).max
			closest_centroid = None

			# Find the closest centroid to data_i
			for j in range(k):
				dist = np.linalg.norm(centroids[j] - data[i])

				if (dist < closest_dist):
					closest_dist = dist
					closest_centroid = j

			# Update that centroid
			centroids[closest_centroid] = centroids[closest_centroid] + (lam * (data[i] - centroids[closest_centroid]))

		# Decrease the learning rate
		lam -= lam_dec
		print('Lambda: {0}'.format(lam))

	return centroids

# Weights for a 2D SOM
def som_init_weights(num_weights, data_size=784, max_val=255):
	weight_vec = np.random.rand(num_weights, num_weights, data_size)
	weight_vec = weight_vec * max_val

	return weight_vec

# Strength calc for weight updates
def som_str(dist, rad):
	return np.exp(-dist / (2 * (rad * rad)))


def dec_rad(rad, it, amt):
	return rad * np.exp(-it / amt)

def dec_lam(lam, it, num_iter):
	return lam * np.exp(-it / float(num_iter))

# 2D SOM
def som(rdata, num_weights=10, ilam=0.001, iradius=8, num_iter=2000):
	# Initialize weights
	weights = som_init_weights(num_weights)

	# Normalize data
	#data = rdata / rdata.max()
	data = rdata

	for i in range(num_iter):
		# Decrease the learning rate & radius
		lam = dec_lam(ilam, i, num_iter)
		radius = dec_rad(iradius, i, num_iter/np.log(iradius))
		print('Lambda: {0}   Radius: {1}'.format(lam, radius))

		# pick a random vector
		vec = data[rnd.randint(0,data.shape[0]-1)]
		
		# Pick a weight as the winner for this data point
		dist_winner = np.finfo(np.float32).max
		indx_winner = -1
		for x in range(num_weights):
			for y in range(num_weights):
				dist = np.linalg.norm(weights[x][y] - vec)

				# New winner found
				if dist < dist_winner:
					dist_winner = dist
					indx_winner = (x,y)

		# Update the weights
		for x in range(num_weights):
			for y in range(num_weights):
				# Only update if within neighbourhood radius
				dist = ((x - indx_winner[0])**2) + ((y - indx_winner[1])**2)
				if dist <= radius**2:
					str = som_str(dist, radius)
					w = weights[x][y] + (lam * str * (vec - weights[x][y]))

					weights[x][y] = w

	return weights

def prt(out_vec, floor=0):
	for i in range(28):
		s = ""
		for j in range(28):
			s += ' ' if out_vec[(i*28)+j] <= floor else '1'
			s += " "
		print(s)


centsom = som(x)

for i in range(10):
	for j in range(10):
		prt(centsom[i][j], 50)


# Need at least 3 centroids, 5 seems best
cents = kmeans(x,3)