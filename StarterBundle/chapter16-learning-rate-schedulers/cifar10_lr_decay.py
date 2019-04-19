# USAGE
# python cifar10_lr_decay.py --output lr_decay_f0.5_plot.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use('Agg')

from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse

def step_decay(epoch):
    # initialize the base initial learning rate, drop factor and epochs to drop every
    init_alpha = 0.01
    factor = 0.5
    drop_every = 5

    # compute learning rate for current epoch
    alpha = init_alpha * (factor ** np.floor((1+epoch)/drop_every))

    return alpha

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output loss/accuracy plot')
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

# define the set of callbacks to be passed to the model during training
callbacks = [LearningRateScheduler(step_decay)]

# initialize optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=2)

# evaluate network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

# plot the training/ test loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label='Training loss')
plt.plot(H.history['val_loss'], label='Validation loss')
plt.plot(H.history['acc'], label='Training accuracy')
plt.plot(H.history['val_acc'], label='Validation accuracy')
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args['output'])