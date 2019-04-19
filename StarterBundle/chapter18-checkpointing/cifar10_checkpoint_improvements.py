# USAGE
# python cifar10_checkpoint_improvements.py --weights weights/improvements
from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
import argparse
import os

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=True, help='path to weights directory')
args = vars(ap.parse_args())

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
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# construct callback to save only the **best** model to disk based on validation loss
fname = os.path.sep.join([args['weights'], "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor='val_loss', verbose=1, save_best_only= True, mode='min')
callbacks = [checkpoint]

# train the network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=2)