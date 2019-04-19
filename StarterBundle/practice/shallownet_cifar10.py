from pyimagesearch.nn.conv import ShallowNet
from keras.optimizers import SGD
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse

# load cifar10
print('[INFO] loading cifar10...')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')/255.
testX = testX.astype('float')/255.

# convert labels to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize labels names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# initialize optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=len(label_names))
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

# training network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=40, verbose=2)

# evaluate network
print('[INFO] evaluating network...')
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=label_names))

# plot the loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label='Training loss')
plt.plot(H.history['val_loss'], label='Validation loss')
plt.plot(H.history['acc'], label='Training accuracy')
plt.plot(H.history['val_acc'], label='Validation accuracy')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('shallownet_cifar10_40_epoch.png')