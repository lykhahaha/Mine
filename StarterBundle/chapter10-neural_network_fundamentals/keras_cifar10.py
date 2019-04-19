# USAGE
# python keras_cifar10.py --output output/keras_cifar10.png
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output image')
args = vars(ap.parse_args())

# Get MNIST dataset
print('[INFO] loading full Cifar10 dataset')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')/255
testX = testX.astype('float')/255
trainX = trainX.reshape((len(trainX), -1))
testX = testX.reshape((len(testX), -1))

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Define neural nets 3072 - 256 - 128 - 10
input_size = trainX.shape[1]
model = Sequential()
model.add(Dense(1024, input_shape=(input_size,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Training neurall networkk
sgd = SGD(lr=0.01)
model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128, verbose=2)

print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label='train_loss')
plt.plot(H.history['val_loss'], label='val_loss')
plt.plot(H.history['acc'], label='train_acc')
plt.plot(H.history['val_acc'], label='val_acc')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])