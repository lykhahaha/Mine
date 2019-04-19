import matplotlib
matplotlib.use('Agg')

from pyimagesearch.nn.conv import MiniVGGNet
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse
import os
import matplotlib.pyplot as plt

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output directory')
ap.add_argument('-m', '--models', required=True, help='path to models directory')
ap.add_argument('-n', '--num-models', type=int, default=5, help='# of models to train')
args = vars(ap.parse_args())

# load dataset and scale it to range [0, 1]
print('[INFO] loading cifar10 dataset...')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')/255.
testX = testX.astype('float')/255.

# convert labels to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize label names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# construct data augmentation for image generator
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# loop over the models
for i in range(args['num_models']):
    print(f"Training model {i+1}/{args['num_models']}...")
    # initialize optimizer and network
    opt = SGD(momentum=0.9, decay=0.01/40, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=len(label_names))
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # train the network
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64), validation_data=(testX, testY), steps_per_epoch=len(trainX)//64, epochs=40, verbose=2)

    # save model to disk
    model.save(os.path.sep.join(args['models'], f'model_{i+1}.model'))

    # evaluate the network
    preds = model.predict(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=label_names)

    # save report to disk
    f = open(os.path.sep.join([args['output'], f'model_{i+1}.txt']), 'w')
    f.write(report)
    f.close()

    # save plot to disk
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(H.history['loss'], label='train_loss')
    plt.plot(H.history['val_loss'], label='val_loss')
    plt.plot(H.history['acc'], label='train_acc')
    plt.plot(H.history['val_acc'], label='val_acc')
    plt.title(f'Training loss/accuracy for model {i+1}')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(os.path.sep.join([args['output'], f'model_{i+1}.png']))
    plt.close()