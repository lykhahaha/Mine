import matplotlib
matplotlib.use('Agg')

from pyimagesearch.nn.conv import MiniVGGNet
from pyimagesearch.callbacks import TrainingMonitor
from keras.optimizers import SGD
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output directory')
args = vars(ap.parse_args())

# get information for process id
print(f'[INFO] process ID: {os.getpid()}')

# load dataset and scale it to range [0, 1]
print('[INFO] loading dataset...')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')/255
testX = testX.astype('float')/255

# initialize label names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# convert it to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize optimizer and network
print('[INFO] compiling the network...')
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=len(label_names))
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

# construct callbacks to be passed to the network
fig_path = os.path.sep.join([args['output'], f'{os.getpid()}.png'])
json_path = os.path.sep.join([args['output'], f'{os.getpid()}.json'])
callbacks = [TrainingMonitor(fig_path, json_path)]

# training the network
print('[INFO] training the network...')
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=100, verbose=2, callbacks=callbacks)