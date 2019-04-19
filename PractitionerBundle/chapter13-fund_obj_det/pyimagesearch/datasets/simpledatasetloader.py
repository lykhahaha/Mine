import cv2
import numpy as np
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		self.preprocessors = preprocessors

		if self.preprocessors == None:
			self.preprocessors = []

	def load(self, img_paths, verbose=-1):
		# data_size, img_size = len(img_paths), cv2.imread(img_paths[0]).shape
		# data = np.zeros((data_size, *img_size))
		data, labels = [], []

		for ii, img_path in enumerate(img_paths):
		    # load the image and extract the class label assuming
		    # that our path has the following format:
		    # /path/to/dataset/{class}/{image}.jpg
		    image = cv2.imread(img_path)
		    label = img_path.split(os.path.sep)[-2]

		    # check to see if our preprocessors are not None
		    if self.preprocessors is not None:
		    	for p in self.preprocessors:
		    		image = p.preprocess(image)

		    # treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
		    data.append(image)
		    labels.append(label)

		    # show an update every `verbose` images
		    if verbose > 0 and ii > 0 and (ii + 1) % verbose == 0:
		    	print(f'[INFO] processed {ii+1}/{len(img_paths)}')

		# return a tuple of the data and labels
		return np.array(data), np.array(labels)