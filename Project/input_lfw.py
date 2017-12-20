import os
import tensorflow as tf
import numpy as np
from keras.utils import np_utils

def add_tuple_to_lists(lists, dir, files):
	labels_train = lists[0]
	labels_test = lists[1]
	data_train = lists[2]
	data_test = lists[3]

	if len(files) > 2:
		for i in range(len(files)-1):
			if (os.path.isfile(os.path.join(dir, files[i]))):
				labels_train.append(os.path.basename(dir))
				data_train.append(os.path.join(dir, files[i]))
		
		if (os.path.isfile(os.path.join(dir, files[i]))):
			labels_test.append(os.path.basename(dir))
			data_test.append(os.path.join(dir, files[len(files)-1]))

def get_labels_and_files(lfw_dir):
	labels_train = []
	labels_test = []
	data_train = []
	data_test = []

	os.path.walk(lfw_dir, add_tuple_to_lists, (labels_train,labels_test,data_train,data_test))

	label_lookup = {}
	it = 0
	dct = {}
	
	for i in range(len(labels_train)):
		lbl = labels_train[i]

		if not lbl in dct:
			dct[lbl] = it
			it += 1

		labels_train[i] = dct[lbl]
		label_lookup[labels_train[i]] = lbl

	for i in range(len(labels_test)):
		labels_test[i] = dct[labels_test[i]]

	labels_train = np_utils.to_categorical(labels_train, len(label_lookup))
	labels_test = np_utils.to_categorical(labels_test, len(label_lookup))

	return (data_train, labels_train), (data_test, labels_test), label_lookup

# https://www.tensorflow.org/programmers_guide/datasets#decoding_image_data_and_resizing_it
#
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [64, 64])
  return image_resized, label

def get_dataset(lfw_dir):
	train_set, test_set, label_lookup = get_labels_and_files(lfw_dir)

	tr_data   = tf.constant(train_set[0])
	tr_labels = tf.constant(train_set[1])
	tr_dataset = tf.data.Dataset.from_tensor_slices((tr_data, tr_labels))
	tr_dataset = tr_dataset.map(_parse_function)

	te_data   = tf.constant(test_set[0])
	te_labels = tf.constant(test_set[1])
	te_dataset = tf.data.Dataset.from_tensor_slices((te_data, te_labels))
	te_dataset = te_dataset.map(_parse_function)

	return tr_dataset, te_dataset, label_lookup

def get_data(lfw_dir, shuffle=True):
	tr_dataset, te_dataset, label_lookup = get_dataset(lfw_dir)
	
	return tr_dataset, te_dataset, label_lookup
