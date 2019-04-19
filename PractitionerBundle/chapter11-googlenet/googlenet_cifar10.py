# USAGE
# python googlenet_cifar10.py --output output --model output/minigooglenet_cifar10.hdf5
import matplotlib
matplotlib.use('Agg')

from pyimagesearch.nn.conv import MiniGoogLeNet
from pyimagesearch.callbacks import TrainingMonitor
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
import argparse

# define total number of epochs along with the initial learning rate
NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate and power pf the polynomial
    max_epochs = NUM_EPOCHS
    base_lr = INIT_LR
    power = 1.0

    # compute new learning rate bse on polynomial decay
    alpha = base_lr * (1 - (epoch/max_epochs))**power

    return alpha

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='path to output model')
ap.add_argument('-o', '--output', required=True, help='path to output directory (logs, plots)')
args = vars(ap.parse_args())

# load the training and testing data
print('[INFO] loading cifar10 data...')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')
testX = testX.astype('float')

# apply mean subtraction to data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert label to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# construct set of callbacks
fig_path = os.path.sep.join([args['output'], f'{os.getpid()}.png'])
json_path = os.path.sep.join([args['output'], f'{os.getpid()}.json'])
callbacks = [TrainingMonitor(fig_path, json_path=json_path), LearningRateScheduler(poly_decay)]

# initialize optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=len(lb.classes_))
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
print('[INFO] training the network...')
model.fit_generator(aug.flow(trainX, trainY, batch_size=64), validation_data=(testX, testY), epochs=NUM_EPOCHS, verbose=2, callbacks=callbacks, steps_per_epoch=len(trainX)//64)

# save the network to disk
print('[INFO] serializing network...')
model.save(args['model'])