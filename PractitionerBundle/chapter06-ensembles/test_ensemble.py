# USAGE
# python test_ensemble.py --models models
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os

# Construct argument parser and parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--models', required=True, help='path to models directory')
args = vars(ap.parse_args())

# load the test data, the nscale it to range [0, 1]
testX, testY = cifar10.load_data()[1]
testX = testX.astype('float')/255.

# initialize label names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Convert labels to vector
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

# Construct the path used to collect the models then initialize the models list
model_paths = os.path.sep.join([args['models'], '*.model'])
model_paths = list(glob.glob(model_paths))
models = []

# loop over model paths, load model and add it to list of models
for i, model_path in enumerate(model_paths):
    print(f'[INFO] loading models {i+1}/{len(model_paths)}...')
    models.append(load_model(model_path))

# initialize list of predictions
print('[INFO] evaluating ensemble...')
predictions = [] # shape (5) - (10000, 10)

# loop over models
for model in models:
    # use current model to make predictions on testing data, then store these predictions in aggregate predictions list
    predictions.append(model.predict(testX, batch_size=64))

# average probabilities across all model predictions, then show a classification report
predictions = np.average(predictions, axis=0) # predictions from list to np.array (5, 10000, 10), average first dim to (10000, 10)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))