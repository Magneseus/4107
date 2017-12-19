import os
import tensorflow as tf
import numpy as np

def add_tuple_to_lists(lists, dir, files):
	labels = lists[0]
	data   = lists[1]

	if len(files) > 1:
		for f in files:
			if (os.path.isfile(os.path.join(dir, f))):
				labels.append(os.path.basename(dir))
				data.append(os.path.join(os.path.abspath(dir), f))

def get_labels_and_files(lfw_dir):
	labels = []
	data   = []

	os.path.walk(lfw_dir, add_tuple_to_lists, (labels,data))

	label_lookup = {}
	it = 0
	dct = {}
	
	for i in range(len(labels)):
		lbl = labels[i]

		if not lbl in dct:
			dct[lbl] = it
			it += 1

		labels[i] = dct[lbl]
		label_lookup[labels[i]] = lbl

	labels2 = []
	for x in labels:
		labels2.append([0 if x != i else 1 for i in range(len(label_lookup))])

	print("test")

	return labels2, data, label_lookup

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
	labels, data, label_lookup = get_labels_and_files(lfw_dir)
	size = len(labels)

	tf_labels = tf.constant(labels)
	tf_data   = tf.constant(data)
	dataset = tf.data.Dataset.from_tensor_slices((tf_data, tf_labels))
	dataset = dataset.map(_parse_function)

	return dataset, size, labels, label_lookup

def split_dataset(dataset, size, split_point, shuffle):
	split_marker = np.floor(size * split_point)

	_data_1 = dataset.take(split_marker)
	_data_2 = dataset.skip(split_marker)

	if shuffle:
		_data_1.shuffle(split_marker)
		_data_2.shuffle(size - split_marker)

	return (_data_1, split_marker), (_data_2, size - split_marker)

def get_data(lfw_dir, split_point=0.9, shuffle=True):
	dataset, size, labels, label_lookup = get_dataset(lfw_dir)
	return split_dataset(dataset, size, split_point, shuffle), label_lookup


((d1,s1),(d2,s2)),ld = get_data('./lfw')
it = d1.make_one_shot_iterator()
ne = it.get_next()
sess = tf.Session()
i = sess.run(ne)