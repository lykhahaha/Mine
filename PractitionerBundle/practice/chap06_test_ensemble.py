from keras.models import load_model
from keras.datasets import cifar10
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import glob
import os
import numpy as np
import argparse

# construct argument parser parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--models', required=True, help='path to models to aggregate')
args = vars(ap.parse_args())

# load test set and initialize the models list
testX, testY = cifar10.load_data()[1]
testX = testX.astype('float')/255.
predictions = []

# convert label to vector
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

# collect all model paths
model_paths = list(glob.glob(os.path.sep.join([args['models'], '*.model'])))

# initialize label names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# loop over the model paths
for model_path in model_paths:
    model = load_model(model_path)
    preds = model.predict(testX, batch_size=64)
    predictions.append(preds)

# average probabilities
predictions = np.average(predictions, axis=0)

# evaluate the ensemble method
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))