from pyimagesearch.nn.conv import ShallowNet
from keras.optimizers import SGD
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.preprocessing import SimplePreprocessor, ImageToArrayPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
args = vars(ap.parse_args())

# grab the image list
print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))

# initialize images preprocessors
sp, iap = SimplePreprocessor(32, 32), ImageToArrayPreprocessor()

# load the dataset and scale raw pixel to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float')/255.

# partition data
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert labels to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

# training network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=2)

# evaluate the network
print('[INFO] evaluating the network...')
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))

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
plt.show()