from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to the input dataset')
args = vars(ap.parse_args())

# get the list of image paths
print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))

# initialize the preprocessors
aap, iap = AspectAwarePreprocessor(width=64, height=64), ImageToArrayPreprocessor()

# load images amd scale it to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float')/255.

# partition data
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert it to vector
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)
target_names = le.classes_

# initialize the optimizer and network
print('[INFO] compiling the network...')
opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(target_names))
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
print('[INFO] training the network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=2)

# evaluate the network
print('[INFO] evaluating the network...')
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=target_names))

# plot the training loss/accuracy plot
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label='train_loss')
plt.plot(H.history['val_loss'], label='val_loss')
plt.plot(H.history['acc'], label='train_acc')
plt.plot(H.history['val_acc'], label='val_acc')
plt.title('Training loss/accuracy without augmentation')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig('flowers17_without_aug.png')