# USAGE
# python keras_mnist.py --output output/keras_mnist.png
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output image')
args = vars(ap.parse_args())

# Get MNIST dataset
print('[INFO] loading full MNIST dataset')
dataset = datasets.fetch_mldata('MNIST original')

# scale the raw pixel intensities to the range [0, 1.0], then
# construct the training and testing splits
data = dataset.data.astype('float')/255
trainX, testX, trainY, testY = train_test_split(data, dataset.target, test_size=0.25)

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Define neural nets 784 - 256 - 128 - 10
input_size = trainX.shape[1]
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

# Training neurall networkk
sgd = SGD(lr=0.01)
model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128, verbose=2)

print('[INFO] evaluating network...')
predictions = model.predict(testX)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

# Plot based on H.history
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