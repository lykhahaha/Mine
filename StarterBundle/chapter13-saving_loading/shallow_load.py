# USAGE
# python shallow_load.py --dataset ../datasets/animals --model shallow_weights.hdf5
from pyimagesearch.preprocessing import SimplePreprocessor, ImageToArrayPreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to pretrained model')
args = vars(ap.parse_args())

# initialize the class labels
class_labels = ['cat', 'dog', 'pandas']

# grab the list of image paths and randomly sample indexes into the image paths list
print('[INFO] sampling images...')
image_paths = np.array(list(paths.list_images(args['dataset'])))
idxs = np.random.randint(0, len(image_paths), size=10)
image_paths = image_paths[idxs]

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset and scale the raw image to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(image_paths)
data = data.astype('float')/255

# load the pre-trained network
print('[INFO] loading pre-trained network...')
model = load_model(args['model'])

# make predictions on the images
print('[INFO] predicting...')
preds = model.predict(data, batch_size=32).argmax(axis=1)

# visualize the results
# loop over the sample images
for ii, image_path in enumerate(image_paths):
    # load the image, draw the prediction and display it to our screen
    image = cv2.imread(image_path)
    cv2.putText(image, f'Label: {class_labels[preds[ii]]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)