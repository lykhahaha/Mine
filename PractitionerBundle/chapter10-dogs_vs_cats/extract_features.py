# USAGE
# python extract_features.py --dataset ../datasets/kaggle_dogs_vs_cats/train --output ../datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5
from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import os
import random
import argparse

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-o', '--output', required=True, help='path to output hdf5 file')
ap.add_argument('-b', '--buffer-size', type=int, default=1000, help='size of feature extraction buffer')
ap.add_argument('-s', '--batch-size', type=int, default=32, help='batch size of image to be passed through feature extractor')
args = vars(ap.parse_args())

# construct batch size for convenience usr
batch_size = args['batch_size']

# grab the list of image paths and extract labels through them
print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))
random.shuffle(image_paths)
labels = [image_path.split(os.path.sep)[-1].split('.')[0] for image_path in image_paths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load model
print('[INFO] loading model...')
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# initialize HDF5 dataset writer
dataset = HDF5DatasetWriter((len(labels), 2048), image_paths, data_key='features', buf_size=args['buffer_size'])
dataset.storeClassLabels(le.classes_)

# construct progress bar
widgets = ['Extracting features: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(labels), widgets=widgets).start()

# loop over batches of images
for i in range(0, len(labels), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    batch_labels = labels[i:i+batch_size]
    batch_images = []

    # loop over images
    for image_path in batch_paths:
        # load image and convert it to Keras-compatitble array
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        # add extra dimension
        image = np.expand_dims(image, axis=0)
        # preprocess by Resnet50
        image = imagenet_utils.preprocess_input(image)
        # add to batch
        batch_images.append(image)

    # pass batch through Resnet50
    batch_images = np.vstack(batch_images)
    features = model.predict(batch_images, batch_size=batch_size)
    features = features.reshape((len(features), -1))
    # add to hdf5 dataset
    dataset.add(features, batch_labels)
    pbar.update(i)

# close dataset
dataset.close()
pbar.finish()