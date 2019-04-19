# USAGE
# python train_model.py --dataset ../datasets/SMILEsmileD --model output/lenet.hdf5
from pyimagesearch.nn.conv import LeNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

# Construct argument parse rand parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset of faces')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# initialize data and labels
data, labels = [], []

# loop over input images
for image_path in sorted(list(paths.list_images(args['dataset']))):
    # load the image, preprocess and store it in list
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28) # (28, 28)
    image = img_to_array(image) # (28, 28, 1)
    data.append(image)

    # get the label by image paths
    label = image_path.split(os.path.sep)[-3]
    label = 'smiling' if label == 'positives' else 'not_smiling'
    labels.append(label)

# scale raw pixel intensities to [0, 1]
data = np.array(data, dtype='float')/255
labels = np.array(labels)

# convert labels from integers to vectors
le = LabelEncoder().fit(labels) # LabelBinarizer only applied for ['a', 'b', 'c', 'a'] or [1, 2, 3] to generate one-hot, for longer, e.g ['pos', 'neg', 'pos', 'neg', 'pos', 'neg'] we need 2 steps
labels = np_utils.to_categorical(le.transform(labels), num_classes=len(le.classes_))

# account for skew in labeled data
class_totals = labels.sum(axis=0) # sum all value in one column
class_weight = np.max(class_totals) / class_totals # [1, 2.56]

# partition data into training and testing splits using 80%, 20%
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# initialize optimizer and model
print('[INFO] compiling model...')
model = LeNet.build(width=28, height=28, depth=1, classes=len(le.classes_))
# opt = SGD(lr=0.02, decay=0.02/15, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=class_weight, batch_size=64, epochs=15, verbose=2)

# evaluating network
print('[INFO] evaluating network...')
predictions = model.predict(testX)
print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1), target_names=[str(c) for c in (le.classes_)]))

# save model to disk
print('[INFO] saving model...')
model.save(args['model'])

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
plt.savefig('output/smile_detect.png')