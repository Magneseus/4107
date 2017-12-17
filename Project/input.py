import os
import tensorflow as tf

def add_tuple_to_lists(lists, dir, files):
	labels = lists[0]
	data   = lists[1]

	for f in files:
		if (os.path.isfile(os.path.join(dir, f))):
			labels.append(os.path.basename(dir))
			data.append(os.path.join(os.path.abspath(dir), f))

def get_labels_and_files(lfw_dir):
	labels = []
	data   = []
	os.path.walk(lfw_dir, add_tuple_to_lists, (labels,data))

	return labels, data

# https://www.tensorflow.org/programmers_guide/datasets#decoding_image_data_and_resizing_it
#
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [128, 128])
  return image_resized, label

def get_dataset(lfw_dir):
	labels, data = get_labels_and_files(lfw_dir)

	tf_labels = tf.constant(labels)
	tf_data   = tf.constant(data)

	dataset = tf.data.Dataset.from_tensor_slices((tf_data, tf_labels))
	dataset = dataset.map(_parse_function)

	return dataset

dataset = get_dataset('./lfw')