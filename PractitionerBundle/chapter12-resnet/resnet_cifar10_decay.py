# USAGE
# python resnet_cifar10_decay.py --output output --model output/resnet_cifar10.hdf5
import matplotlib
matplotlib.use('Agg')

from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import TrainingMonitor
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import argparse
import os
import sys
from keras.models import Sequential

# set a high recursion limit so THeano doesn't complain
sys.setrecursionlimit(5000)

# define the total number of epochs to train for along with the initial learning rate
MAX_EPOCH = 100
INIT_LR = 1e-1

def poly_decay(epoch):
    # initialize maximum number of epochs, base learning rate and power of polynomial
    max_epoch = MAX_EPOCH
    base_lr = INIT_LR
    power = 1.0

    # compute new learning rate based on polynomial decay
    alpha = base_lr * (1 - epoch / float(max_epoch))**power

    return alpha

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='path to output model')
ap.add_argument('-o', '--output', required=True, help='path to output directory (logs, plots)')
args = vars(ap.parse_args())

# load training and testing data, convert images to float
print('[INFO] loading cifar10...')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')
testX = testX.astype('float')

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert labels to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct data generator for image augmentation
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# construct callbacks
json_path = os.path.sep.join([args['output'], f'{os.getpid()}.json'])
fig_path = os.path.sep.join([args['output'], f'{os.getpid()}.png'])
callbacks = [TrainingMonitor(fig_path, json_path=json_path), LearningRateScheduler(poly_decay)]

# initialize optimizer and model
print('[INFO] compiling mode...')
model = ResNet.build(width=32, height=32, depth=3, classes=len(lb.classes_), stages=[9, 9, 9], filters=[64, 64, 128, 256], reg=5e-4)
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
print('[INFO] training the network...')
model.fit_generator(aug.flow(trainX, trainY, 128), validation_data=(testX, testY), epochs=100, verbose=2, callbacks=callbacks, steps_per_epoch=len(trainX)//128)

# save the network to disk
print('[INFO] serializing model...')
model.save(args['model'])