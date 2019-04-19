from pyimagesearch.nn.conv import LeNet
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import imutils

# Construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to output model to save')
args = vars(ap.parse_args())

# initialize the data and labels list
data, labels = [], []

for image_path in sorted(list(paths.list_images(args['dataset']))):
    # load, pre-process and store it to data list
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    # extract label from image path
    label = image_path.split(os.path.sep)[-3]
    labels.append('smiling' if label=='positives' else 'not smiling')

# scale data to range [0, 1]
data = np.array(data, dtype='float')/255.

# convert labels to vector
le = LabelEncoder()
labels = to_categorical(le.fit_transform(labels), 2)

# handle skew of label
class_totals = labels.sum(axis=0)
class_weight = class_totals.max()/class_totals

# partition the data
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# initialize the network
print('[INFO] compiling the network...')
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the network
print('[INFO] training the network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=15, verbose=2, class_weight=class_weight)

# evaluate the network
print('[INFO] evaluating the network...')
preds = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=le.classes_))

# save model to disk
print('[INFO] saving model to disk...')
model.save(args['model'])

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label='train_loss')
plt.plot(H.history['val_loss'], label='val_loss')
plt.plot(H.history['acc'], label='train_acc')
plt.plot(H.history['val_acc'], label='val_acc')
plt.title('Training loss and accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig('smile_epoch_15.png')