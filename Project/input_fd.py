import os
import numpy as np
from scipy import ndimage
from keras.utils.np_utils import to_categorical

# https://stackoverflow.com/questions/35723865/read-a-pgm-file-in-python
def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    assert pgmf.readline() == 'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    
    return ndimage.zoom(raster, (64.0/height, 64.0/width))

def get_data(directory):
	labels = np.array([])
	data = []

	for filename in os.listdir(directory):
		path = os.path.join(directory, filename)

		label = filename.split('.')[0]
		label = int(label[-2:]) - 1

		with open(path) as f:
			image = read_pgm(f)

		labels = np.concatenate((labels, [label]))
		data.append(image)

	labels = to_categorical(labels)
	data = np.array(data)

	labels = labels.astype(np.float32)
	data = data.astype(np.float32)

	return data, labels
