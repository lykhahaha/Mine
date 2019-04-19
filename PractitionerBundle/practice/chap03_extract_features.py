from keras.applications import VGG16, imagenet_utils
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import progressbar
import argparse
import numpy as np
import random
import os

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-o', '--output', required=True, help='path to output HDF5')
ap.add_argument('-b', '--buffer-size', type=int, default=1000, help='size of buffer')
ap.add_argument('-s', '--batch-size', type=int, default=32, help='batch size of image to be passed through network')
args = vars(ap.parse_args())

# grab the list of image paths and shuffle the list
print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))
random.shuffle(image_paths)

# extract labels from image paths
labels = [image_path.split(os.path.sep)[-2] for image_path in image_paths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# store batch-size for convenience
batch_size = args['batch_size']

# load VGG16
print('[INFO] loading VGG16...')
model = VGG16(weights='imagenet', include_top=False)

# initialize the HDF5 dataset writer
dataset = HDF5DatasetWriter(args['output'], (len(image_paths), 7*7*512), data_key='features', buf_size=args['buffer_size'])
dataset.storeClassLabels(le.classes_)

# construct progressbar
widgets = ['Extracting features ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    batch_labels = labels[i:i+batch_size]
    batch_images = []

    for image_path in batch_paths:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        # image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batch_images.append(image)

    batch_images = np.array(batch_images)
    features = model.predict(batch_images, batch_size=batch_size)

    features = features.reshape((len(features), np.prod(features.shape[1:])))

    dataset.add(features, batch_labels)

    pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()