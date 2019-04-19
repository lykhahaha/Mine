# USAGE
# python train_models.py --output output --models models
import matplotlib
matplotlib.use('Agg')

from pyimagesearch.nn.conv import MiniVGGNet
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Construct an argument parser and parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output directory')
ap.add_argument('-m', '--models', required=True, help='path to output models directory')
ap.add_argument('-n', '--num-models', type=int, default=5, help='# of models to train')
args = vars(ap.parse_args())

# load dataset and scale it to range [0, 1]
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')/255.
testX = testX.astype('float')/255.

# Convert labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize label names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Construct image generator for data augmentation
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# loop over number of models to train
for i in range(0, args['num_models']):
    # initialize optimizer and compile model
    print(f"[INFO] training model {i+1}/{args['num_models']}...")
    opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(32, 32, 3, len(label_names))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # train network
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64), validation_data=(testX, testY), steps_per_epoch=len(trainX)//64, epochs=40, verbose=2)

    # save model to disk
    model.save(os.path.sep.join([args['models'], f'model_{i}.model']))

    # evaluate model
    preds = model.predict(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=label_names)

    # save classification report  to file
    p = [args['output'], f'model_{i}.txt']
    f = open(os.path.sep.join(p), 'w')
    f.write(report)
    f.close()

    # plot accuracy/loss
    p = [args['output'], f'model_{i}.png']
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(H.history['loss'], label='train_loss')
    plt.plot(H.history['val_loss'], label='val_loss')
    plt.plot(H.history['acc'], label='acc')
    plt.plot(H.history['val_acc'], label='val_acc')
    plt.title(f'Training Loss and Accuracy for model {i}')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(os.path.sep.join(p))
    plt.close()