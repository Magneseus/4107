import os
import argparse
import numpy as np
import tensorflow as tf
from input_fd import get_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=256,
					help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='./faces',
					help='Path to the face data directory.')

parser.add_argument('--num_epochs', type=int, default=100,
					help='Number of training epochs between testing.')

parser.add_argument('--global_step', type=int, default=0,
					help='Current step to start at. (If non-zero, will attempt to load a checkpoint)')

FLAGS = parser.parse_args()

# Get face dataset
data, labels = get_data(FLAGS.data_dir)

# PARAMETERS
NUM_OUTPUTS = len(labels[0])
SIZE_INPUT_X = data.shape[1]
SIZE_INPUT_Y = data.shape[2]
PERCENT_TRAIN = 0.7

# Split the raw data into training and testing data
split_marker = int(np.floor(len(data) * PERCENT_TRAIN))

train_data = data[0:split_marker]
train_labels = labels[0:split_marker]

test_data = data[split_marker:]
test_labels = labels[split_marker:]



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
def model(features, labels):
	input_layer = tf.reshape(features, [-1, SIZE_INPUT_X, SIZE_INPUT_Y, 1])

	with tf.variable_scope('conv1') as scope:
		weights = get_weight(
				'weights',
				[5, 5, 1, 64],
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

		conv = tf.nn.conv2d(pool1, weights, [1,1,1,1], padding='SAME')
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

		conv = tf.nn.conv2d(pool2, weights, [1,1,1,1], padding='SAME')
		biases = get_bias('biases', [64], tf.constant_initializer(0.2))
		bias_conv = tf.nn.bias_add(conv, biases)

		conv3 = tf.nn.relu(bias_conv, name=scope.name)

	with tf.variable_scope('pool3') as scope:
		pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
				strides=[1, 2, 2, 1], padding='SAME', name='pool3')
	'''

	with tf.variable_scope('fc') as scope:
		pool2_shape = pool2.get_shape()
		reshape = tf.reshape(pool2, [-1, (pool2_shape[1] * pool2_shape[2] * pool2_shape[3]).value])
		
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
				[600, NUM_OUTPUTS],
				1/600.0)
		biases = get_bias('biases', [NUM_OUTPUTS], tf.constant_initializer(0.0))

		output = tf.add(tf.matmul(fc, weights), biases, name=scope.name)

	return output

X = tf.placeholder("float", [None, SIZE_INPUT_X, SIZE_INPUT_Y])
Y = tf.placeholder("float", [None, NUM_OUTPUTS])

output = model(X, Y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(output, 1)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


with tf.Session() as sess:
	# Initialize variables
	tf.global_variables_initializer().run()

	acc_string = ""

	# Restore checkpoint
	if FLAGS.global_step != 0:
		print("Restoring model")
		saver.restore(sess, "./checkpoints/fd/model_fd-{}".format(FLAGS.global_step))

	print("Start training")

	for epoch in range(FLAGS.global_step+1, FLAGS.global_step + 1 + FLAGS.num_epochs):
		sess.run(train_op, feed_dict={X: train_data, Y: train_labels})

		save_path = saver.save(sess, './checkpoints/fd/model_fd', global_step=epoch)
		print("Saved checkpoint: %s" % save_path)

		run_acc = np.mean(np.argmax(test_labels, axis=1) ==
				sess.run(predict_op, feed_dict={X: test_data}))
		print(epoch, run_acc)
		acc_string += "{}\n".format(run_acc)

		# End of one test/train cycle

	# End of entire run cycle
	print(acc_string)