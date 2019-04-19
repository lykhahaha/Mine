from pyimagesearch.nn.conv import ShallowNet
from keras.optimizers import SGD
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths

# Construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# grab the images list
print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))

# initialize preprocessors
sp, iap = SimplePreprocessor(32, 32), ImageToArrayPreprocessor()

# load dataset and scale raw image to range [0, 1]
sdl = SimpleDatasetLoader([sp, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float')/255

# partition data
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert labels to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=len(lb.classes_))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=2)

# save model to disk
print('[INFO] saving model...')
model.save(args['model'])

# evaluate network
print('[INFO] evaluating network...')
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))

# plt the training/test loss and accuracy
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
plt.show()