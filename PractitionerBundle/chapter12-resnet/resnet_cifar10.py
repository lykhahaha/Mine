# USAGE
# python resnet_cifar10.py --checkpoints output/checkpoints
# python resnet_cifar10.py --checkpoints output/checkpoints --model output/checkpoints/epoch_50.hdf5 --start-epoch 50
import matplotlib
matplotlib.use('Agg')

from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import EpochCheckpoint, TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
import numpy as np
import argparse
import sys
from keras.models import Sequential

# set a high recursion limit so Theano doesn't complain
sys.setrecursionlimit(5000)

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--checkpoints', required=True, help='path to output checkpoint directory')
ap.add_argument('-m', '--model', type=str, help='path to specific model checkpoint to load')
ap.add_argument('-s', '--start-epoch', type=int, default=0, help='epoch to restart training at')
args = vars(ap.parse_args())

# load training and testing data, convert data to float
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')
testX = testX.astype('float')

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert label to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct data augmentation for image generator
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# if there is no specific model checkpoint supplied, initialize the network (ResNet-56) and compile the model
if args['model'] is None:
    print('[INFO] compiling model...')
    opt = SGD(lr=1e-2, momentum=0.9)
    model = ResNet.build(width=32, height=32, depth=3, classes=len(lb.classes_), stages=(9, 9, 9), filters=(16, 64, 128, 256), reg=1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
else:
    print(f"[INFO] loading {args['model']}")
    model = load_model(args['model'])

    # update learning rate
    print(f'[INFO] old learning rate: {K.get_value(model.optimizer.lr)}')
    K.set_value(model.optimizer.lr, 1e-2)
    print(f'[INFO] old learning rate: {K.get_value(model.optimizer.lr)}')

# construct set of callbacks
callbacks = [
    EpochCheckpoint(args['checkpoints'], every=5, start_at=args['start_epoch']),
    TrainingMonitor('output/resnet56_cifar10.png', 'output/resnet56_cifar10.json', start_at=args['start_epoch'])
]

# train the network
print('[INFO] training the network...')
model.fit_generator(aug.flow(trainX, trainY, 128), validation_data=(testX, testY), epochs=100, verbose=2, callbacks=callbacks, steps_per_epoch=len(trainX)//128)

