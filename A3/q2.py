import numpy as np
import random
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scikitplot as skplt

# Get the MNIST data, if not already there
DATA_DIR = "./"
mnist = fetch_mldata('MNIST original', data_home=DATA_DIR)

# Initiate the random generator
rnd = random.Random()

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
def som(rdata, num_weights=50, ilam=0.005, iradius=30, num_iter=5000):
	# Initialize weights
	weights = som_init_weights(num_weights)

	# Normalize data
	data = rdata.astype(np.float32) / rdata.max()
	#data = rdata

	for i in range(num_iter):
		# Decrease the learning rate & radius
		lam = dec_lam(ilam, i, num_iter)
		radius = dec_rad(iradius, i, num_iter/np.log(iradius))
		if i%10 == 0: print('Lambda: {0}   Radius: {1}'.format(lam, radius))

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

# Generate a map of the average distance between weights in the SOM
def som_avg_dist(som):
	avg_dist = np.zeros(shape=(som.shape[0], som.shape[1]))

	for x in range(som.shape[0]):
		for y in range(som.shape[1]):
			w = som[x][y]

			dist = 0.0
			num = 0
			# Calculate avg dist to neighbours
			for r in range(-1,2):
				for r2 in range(-1,2):
					x2 = x + r
					y2 = y + r2
					if (x2 >= 0 and x2 < som.shape[0] and y2 >= 0 and y2 < som.shape[1]):
						dist += np.linalg.norm(w - som[x2][y2])
						num += 1

			avg_dist[x][y] = dist / float(num)

	# Normalize distances
	return avg_dist / avg_dist.max()

# Print out a digit
def prt(out_vec, floor=0):
	for i in range(28):
		s = ""
		for j in range(28):
			s += ' ' if out_vec[(i*28)+j] <= floor else '1'
			s += " "
		print(s)

# Classify a digit
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




################ Self-Organizing Map #####################
centroids_som = som(x)

# Plot the distance avg of the SOM
'''
normalizedSOM = som_avg_dist(centroids_som)
plt.imshow(normalizedSOM)
plt.colorbar()
plt.show()
'''

################ K-Means Clustering #####################
# Need at least 3 centroids, 5 seems best
num_centroids = 5
cents = kmeans(x, num_centroids)

# Classifying the clusters
'''
y = [0 for i in range(num_centroids)]
for i in range(num_centroids):
	y[i] = get_smallest_distance(cents[i])
'''

# Plotting the dimension-reduced data
'''
pca = PCA(random_state=1)
pca.fit(cents)
skplt.decomposition.plot_pca_2d_projection(pca, cents, y)
plt.show()
'''