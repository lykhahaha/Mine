from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
import os
import argparse

# Construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=True, help='path to best weights file')
args = vars(ap.parse_args())

# load the dataset and scale it to vector
print('[INFO] loading dataset...')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')/255
testX = testX.astype('float')/255

# convert label to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize optimizer and network
print('[INFO] compiling the network...')
opt = SGD(lr=0.01, momentum=0.9, decay=0.01/40, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

# set callback to save the best model weights
callbacks = [ModelCheckpoint(args['weights'], verbose=1, save_best_only=True, mode='min')]

# training the network
print('[INFO] training the network...')
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=2, callbacks=callbacks)