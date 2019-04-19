from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.preprocessing import SimplePreprocessor, ImageToArrayPreprocessor
from keras.models import load_model
import cv2
import numpy as np
import argparse
from imutils import paths

# Construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# Grab the image list and get randomly 10 images for testing
print('[INFO] loading randomly 10 images for testing...')
image_paths = np.array(list(paths.list_images(args['dataset'])))
idxes = np.random.randint(0, len(image_paths), 10)
image_paths = image_paths[idxes]

# initialize the preprocessors
sp, iap = SimplePreprocessor(32, 32), ImageToArrayPreprocessor()

# load 10 images and scale it to vector
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(image_paths)
data = data.astype('float')/255
target_names = ['cats', 'dogs', 'pandas']

# loading network and its weight
print('[INFO] loading network and its weight...')
model = load_model(args['model'])

# predict labels for the 10 images
print('[INFO]predicting the 10 images...')
preds = model.predict(data, batch_size=32).argmax(axis=1)

# show prediction and corresponding images
for i, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    cv2.putText(image, f'Label: {target_names[preds[i]]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)