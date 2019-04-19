from pyimagesearch.io import HDF5DatasetWriter
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import os
import progressbar
import random
import argparse

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-o', '--output', required=True, help='path to output HDF5')
ap.add_argument('-b', '--buffer-size', type=int, default=1000, help='size of buffer')
args = vars(ap.parse_args())

# grab the list of image paths and shuffle them
image_paths = list(paths.list_images(args['dataset']))
random.shuffle(image_paths)
buf_size = args['buffer_size']

# get the labels by extracting image paths
labels = [image_path.split(os.path.sep)[-2] for image_path in image_paths]
le = LabelEncoder()
labels = le.fit_transform(labels)
label_names = le.classes_

# construct HDF5 dataset writer
dataset = HDF5DatasetWriter(args['output'], (len(image_paths), 224, 224, 3), buf_size=buf_size)
dataset.storeClassLabels(label_names)

# construct progressbar
widgets = ['Convert images ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
pbar = progressbar.ProgressBar(len(image_paths), widgets=widgets).start()

for i in range(0, len(image_paths), buf_size):
    batch_paths = image_paths[i:i+buf_size]
    batch_labels = labels[i:i+buf_size]
    batch_images = []

    for image_path in batch_paths:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)

        batch_images.append(image)
    
    batch_images = np.array(batch_images)
    batch_images = batch_images.astype('float')/255.
    dataset.add(batch_images, batch_labels)
    pbar.update(i)

dataset.close()
pbar.finish()