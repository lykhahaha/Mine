# USAGE
# python cifar10_monitor.py --output output
# set matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use('Agg')

from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
import os
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output directory')
args = vars(ap.parse_args())

# show information on the process ID
print(f'[INFO] process ID: {os.getpid()}')

# load cifar10 and scale it to [0, 1]
print('[INFO] loading cifar10 data...')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')/255
testX = testX.astype('float')/255

# convert the labels from integer to vector
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# initialize the label names for the CIFAR-10 dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# initialize optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# construct the set of callbacks
fig_path = os.path.sep.join([args['output'], f'{os.getpid()}.png']) # 'output\\125.png'
json_path = os.path.sep.join([args['output'], f'{os.getpid()}.json'])
callbacks = [TrainingMonitor(fig_path, json_path=json_path)]

# train the network
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=100, callbacks=callbacks, verbose=2)