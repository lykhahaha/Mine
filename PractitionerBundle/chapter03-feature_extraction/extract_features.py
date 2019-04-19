# USAGE
# python extract_features.py --dataset ../datasets/animals/images --output ../datasets/animals/hdf5/features.hdf5
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

# construct argument parser and parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-o', '--output', required=True, help='path to output HDF5 file')
ap.add_argument('-b', '--batch-size', type=int, default=32, help='batch size of image to be passed through network')
ap.add_argument('-s', '--buffer-size', type=int, default=32, help='size of feature extraction buffer')
args = vars(ap.parse_args())

# store batch size in another variable
batch_size = args['batch_size']

# grab the list of images that we'll be describing then randomly shuffle them to allow for easy training and testing splits via array slicing during training time
print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))
random.shuffle(image_paths)

# extract class labels from image paths then encode labels
labels = [image_path.split(os.path.sep)[-2] for image_path in image_paths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load VGG16 network
print('[INFO] loading network...')
model = VGG16(weights='imagenet', include_top=False)

# initialize HDF5 dataset writer, then store class label names in dataset
dataset = HDF5DatasetWriter((len(image_paths), 512*7*7), args['output'], data_key='features', buf_size=args['buffer_size'])
dataset.storeClassLabels(le.classes_)

# initialize progress bar
widgets = ['Extracting Features: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()

# loop over image in batches
for i in range(0, len(image_paths), batch_size):
    # extract batch of images and labels, then initialize list of actual images that will be passed through network of feature extraction
    batch_paths = image_paths[i:i+batch_size]
    batch_labels = labels[i:i+batch_size]
    batch_images = []
    # loop over images and labels in current batch
    for j, image_path in enumerate(batch_paths):
        # load input image using Keras helper utility while ensuring image is resized to 224x224 pixels
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        # preprocess image by expanding dimension and subtracting mean RGB pixel intensity by Imagenet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batch_images.append(image)
    # pass images through network and use outputs as our actual features
    batch_images = np.vstack(batch_images)     
    features = model.predict(batch_images, batch_size=batch_size) # shape (batch_size, 7, 7, 512)
    # reshape features so that each image is represented by flattened feature vector of MaxPooling2D outputs
    features = features.reshape((features.shape[0], -1))
    # add features and labels to our HDF5 dataset
    dataset.add(features, batch_labels)
    pbar.update(i)

# close dataset
dataset.close()
pbar.finish()