from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from pyimagesearch.callbacks import TrainingMonitor
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def step_decay(epoch):
    init_alpha = 0.05
    factor = 0.5
    drop_every = 10
    return float(init_alpha * factor**np.floor((1+epoch)/drop_every))

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-o', '--output', required=True, help='path to output for training monitor')
args = vars(ap.parse_args())

# grab the list of image paths
print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))

# initialize the preprocessors
aap, iap = AspectAwarePreprocessor(64, 64), ImageToArrayPreprocessor()

# load images and scale it to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float')/255.

# partition data and label
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert label to vector
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)
target_names = le.classes_

# monitor training
fig_path = os.path.sep.join([args['output'], f'{os.getpid()}.png'])
json_path = os.path.sep.join([args['output'], f'{os.getpid()}.json'])
# callbacks = [TrainingMonitor(fig_path, json_path=json_path)]

# set callback for step decay of learning rate
callbacks = [LearningRateScheduler(step_decay), TrainingMonitor(fig_path, json_path=json_path)]

# construct data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# initialize the optimizer and network
opt = SGD(lr=0.05, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(target_names))
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
print('[INFO] training the network...')
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64), validation_data=(testX, testY), steps_per_epoch=len(trainX)//64, epochs=100, verbose=2, callbacks=callbacks)

# evaluate the network
print('[INFO] evaluating the network...')
preds = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=target_names))

# plot the training loss/accuracy plot
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label='train_loss')
plt.plot(H.history['val_loss'], label='val_loss')
plt.plot(H.history['acc'], label='train_acc')
plt.plot(H.history['val_acc'], label='val_acc')
plt.title('Training loss/accuracy with augmentation')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig('flowers17_with_aug.png')