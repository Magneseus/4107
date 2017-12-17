import os
import tensorflow as tf

def add_tuple_to_lists(lists, dir, files):
	labels = lists[0]
	data   = lists[1]

	for f in files:
		if (os.path.isfile(os.path.join(dir, f))):
			labels.append(os.path.basename(dir))
			data.append(os.path.join(dir, f))

def get_labels_and_files(lfw_dir):
	labels = []
	data   = []
	os.path.walk(lfw_dir, add_tuple_to_lists, (labels,data))

	return labels, data

labels, data = get_labels_and_files("./lfw")