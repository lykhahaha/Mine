# USAGE
# python minivggnet_flowers17_data_aug.py --dataset ../datasets/flowers17
from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct argument parser and parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
args = vars(ap.parse_args())

# grab list of images that we'll describe, then extract thw labels from path
print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))
class_names = [image_path.split(os.path.sep)[-2] for image_path in image_paths]
class_names = [str(x) for x in np.unique(class_names)]

# initialize the image preprocessors
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load dataset from disk then scale raw pixel intensities to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
data, labels = sdl.load(image_paths,verbose=500)
data = data.astype('float')/255.0

# partition data into training and testing splits using 75%, 25%
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert labels from integers to vectors
le = LabelBinarizer().fit(trainY)
trainY = le.transform(trainY)
testY = le.transform(testY)

# construct image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest') # zoom between [0.8, 1.2]

# initialize optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=0.05)
# opt = SGD(lr=0.1, decay=0.1/100)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(class_names))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train network
print('[INFO] training network...')
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), steps_per_epoch=len(trainX)//32, epochs=100, verbose=2) # one epoch will run steps_per_epoch times, each times it runs aug.flow to have 32 images to feed to nn

# evaluate network
print('[INFO] evaluating network...')
predictions = model.predict(testX)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=class_names))

# plot training/test loss and accuracy
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
plt.savefig('output/flowers17_with_aug.png')