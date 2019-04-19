from pyimagesearch.nn.conv import MiniGoogLeNet
from pyimagesearch.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
import os
import argparse
import numpy as np

BASE_LR = 1e-2
MAX_EPOCH = 70

def poly_decay(epoch):
    power = 1.0

    return BASE_LR * (1 - epoch / float(MAX_EPOCH))**power

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output (plot, json)')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# load cifar10 datasets
print('[INFO] loading cifar10...')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')
testX = testX.astype('float')

# normalize data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert labels to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct image generator for image generator
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)

# construct callbacks
fig_path = os.path.sep.join([args['output'], f'{os.getpid()}.png'])
json_path = os.path.sep.join([args['output'], f'{os.getpid()}.json'])
callbacks = [TrainingMonitor(fig_path, json_path), LearningRateScheduler(poly_decay)]

# initialize model and compile
print('[INFO] compiling the model...')
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=len(lb.classes_))
opt = SGD(lr=BASE_LR, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
print('[INFO] train the network...')
model = Sequential()
model.fit(aug.flow(trainX, trainY, 64), validation_data=(testX, testY), epochs=MAX_EPOCH, verbose=2, callbacks=callbacks, steps_per_epoch=len(trainX)//64)

# save model to disk
print('[INFO] serializing model...')
model.save(args['model'], overwrite=True)