import os
import argparse
import numpy as np
import tensorflow as tf
from input import get_data

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=128,
					help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='./lfw',
					help='Path to the LFW data directory.')

parser.add_argument('--num_epochs', type=int, default=1,
					help='Number of training epochs between testing.')

parser.add_argument('--num_runs', type=int, default=5,
					help='Number of train/test cycles.')

FLAGS = parser.parse_args()


# Get LFW dataset
((train_data, train_size), (test_data, test_size)), label_lookup = get_data(FLAGS.data_dir)

train_data = train_data.batch(FLAGS.batch_size)
test_data = test_data.batch(FLAGS.batch_size)

# Define the iterator for the datasets
train_it = train_data.make_initializable_iterator()
train_next = train_it.get_next()

test_it = test_data.make_initializable_iterator()
test_next = test_it.get_next()

# PARAMETERS
NUM_LABELS = len(label_lookup)
SIZE_INPUT = 64

def get_weight(name, shape, stddev_):
	return tf.get_variable(
		name,
		shape,
		initializer=tf.truncated_normal_initializer(stddev=stddev_, dtype=tf.float32),
		dtype=tf.float32)

def get_bias(name, shape, initializer_):
	return tf.get_variable(
		name,
		shape,
		initializer=initializer_,
		dtype=tf.float32)

# Define the network model
def model(input):
	input_layer = tf.reshape(input, [-1, 64, 64, 3])

	with tf.variable_scope('conv1') as scope:
		weights = get_weight(
				'weights',
				[5, 5, 3, 64],
				5e-2)

		conv = tf.nn.conv2d(input_layer, weights, [1,1,1,1], padding='SAME')
		biases = get_bias('biases', [64], tf.constant_initializer(0.0))
		bias_conv = tf.nn.bias_add(conv, biases)

		conv1 = tf.nn.relu(bias_conv, name=scope.name)

	with tf.variable_scope('pool1') as scope:
		pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
				padding='SAME', name='pool1')

	with tf.variable_scope('conv2') as scope:
		weights = get_weight(
				'weights',
				[5, 5, 64, 64],
				5e-2)

		conv = tf.nn.conv2d(conv1, weights, [1,1,1,1], padding='SAME')
		biases = get_bias('biases', [64], tf.constant_initializer(0.1))
		bias_conv = tf.nn.bias_add(conv, biases)

		conv2 = tf.nn.relu(bias_conv, name=scope.name)

	with tf.variable_scope('pool2') as scope:
		pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
				strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	'''
	with tf.variable_scope('conv3') as scope:
		weights = get_weight(
				'weights',
				[5, 5, 64, 64],
				5e-2)

		conv = tf.nn.conv2d(conv2, weights, [1,1,1,1], padding='SAME')
		biases = get_bias('biases', [64], tf.constant_initializer(0.2))
		bias_conv = tf.nn.bias_add(conv, biases)

		conv3 = tf.nn.relu(bias_conv, name=scope.name)

	with tf.variable_scope('pool3') as scope:
		pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
				strides=[1, 2, 2, 1], padding='SAME', name='pool3')
	'''

	with tf.variable_scope('fc') as scope:
		pool2_shape = pool2.get_shape()
		reshape = tf.reshape(pool2, [FLAGS.batch_size, (pool2_shape[1] * pool2_shape[2] * pool2_shape[3]).value])
		
		re_shape = reshape.get_shape()

		weights = get_weight(
			'weights',
			[(re_shape[1]).value, 600],
			0.04)
		biases = get_bias('biases', [600], tf.constant_initializer(0.1))

		fc = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

	with tf.variable_scope('logits') as scope:
		weights = get_weight(
				'weights',
				[600, NUM_LABELS],
				1/600.0)
		biases = get_bias('biases', [NUM_LABELS], tf.constant_initializer(0.0))

		output = tf.add(tf.matmul(fc, weights), biases, name=scope.name)

	return output

X = tf.placeholder("float", [None, SIZE_INPUT, SIZE_INPUT, 3])
Y = tf.placeholder("float", [None, NUM_LABELS])

output = model(X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(output, 1)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
	# Initialize variables
	tf.global_variables_initializer().run()

	print("Start training")

	for run in range(FLAGS.num_runs):
		for epoch in range(FLAGS.num_epochs):
			sess.run(train_it.initializer)
			while True:
				try:
					data, labels = sess.run(train_next)

					sess.run(train_op, feed_dict={X: data, Y: labels})
				except tf.errors.OutOfRangeError:
					break

			# End of one epoch

			save_path = saver.save(sess, './checkpoints/model', global_step=(run*FLAGS.num_epochs)+epoch)
			print("Saved checkpoint: %s" % save_path)

		# End of X epochs
		sess.run(test_it.initializer)
		while True:
			try:
				data, labels = sess.run(test_next)

				print(i, np.mean(np.argmax(labels, axis=1) ==
						sess.run(predict_op, feed_dict={X: data})))

			except tf.errors.OutOfRangeError:
				break

		# End of one test/train cycle

	# End of entire run cycle