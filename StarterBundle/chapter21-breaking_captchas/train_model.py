# USAGE
# python train_model.py --dataset dataset --model output/lenet.hdf5
from pyimagesearch.nn.conv import LeNet
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
from pyimagesearch.utils.captchahelper import preprocess
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input datset')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# initialize data and labels
data, labels = [], []

# loop over input images
for image_path in paths.list_images(args['dataset']):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)

    # extract class label from image path and update labels first
    label = image_path.split(os.path.sep)[-2]
    labels.append(label)

# scale raw pixel to [0, 1]
data = np.array(data, dtype='float')/255
labels = np.array(labels)

# partition data into 75% and 25%
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert labels to vectors
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# initialize optimizer and model
print('[INFO] compiling model...')
model = LeNet.build(width=28, height=28, depth=1, classes=len(le.classes_))
opt = SGD(lr=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=15, verbose=2)

# evaluating network
print('[INFO] evaluating model...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# save model to disk
print('[INFO] serializing network...')
model.save(args['model'])

# plot the training/ test loss and accuracy
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
plt.savefig('output/lenet.png')