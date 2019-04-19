# USAGE
# python shallow_train.py --dataset ../datasets/animals --model shallow_weights.hdf5
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor, ImageToArrayPreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# grab the list of images we use
print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset and scale the raw image to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float')/255

# partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=2)

# save the network to disk
print('[INFO] saving the model...')
model.save(args['model'])

# evaluate the network
print('[INFO] evaluating the network...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=['cat', 'dog', 'pandas']))

# plot the training loss and accuracy
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