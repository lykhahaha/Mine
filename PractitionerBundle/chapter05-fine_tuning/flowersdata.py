# USAGE
# python flowersdata.py --train ../flowers_data/train --test ../flowers_data/valid --model flowers_data.model
from pyimagesearch.nn.conv import FCHeadNet
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import os

# Construct argument parser and parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--train', required=True, help='path to input train dataset')
ap.add_argument('-t', '--test', required=True, help='path to input test dataset')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# Construct image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Grab the list of images and extract class labels for evaluating
print('[INFO] loading train images...')
image_paths = list(paths.list_images(args['train']))
class_names = [image_path.split(os.path.sep)[-2] for image_path in image_paths]
class_names = [str(x) for x in np.unique(class_names)]
print('[INFO] loading test images...')
image_paths_test = list(paths.list_images(args['test']))

# initialize image preprocessors
aap, iap = AspectAwarePreprocessor(224, 224), ImageToArrayPreprocessor()

# load dataset and scale to range[0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
trainX, trainY = sdl.load(image_paths, verbose=500)
trainX = trainX.astype('float')/255
testX, testY = sdl.load(image_paths_test, verbose=500)
testX = testX.astype('float')/255

# convert labels to vector
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# load VGG16, ensuring head FC layer sets are left off
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# initialize new head of network
head_model = FCHeadNet.build(base_model, len(class_names), 256)

# initialize model
model = Model(inputs=base_model.input, outputs=head_model)

# freeze all layers in base model
for layer in base_model.layers:
    layer.trainable = False

# compile model
print('[INFO] compiling model...')
opt = RMSprop(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train head of network (body of network is frozen)
print('[INFO] training head...')
model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), steps_per_epoch=len(trainX)//32, epochs=25, verbose=2)

# evaluate network after initialization
print('[INFO] evaluating after initialization...')
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=class_names))

# Now that head FC is trained/initialized, let's unfreeze final set of CONV layers
for layer in base_model.layers[15:]:
    layer.trainable = True

# due to change of model, we need to re-compile, this time we use SGD
print('[INFO] re-compiling network...')
opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train network, this time fine-tuning final-set of CONV layers along with head FC layer
print('[INFO] fine-tuning model...')
model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), steps_per_epoch=len(trainX)//32, epochs=100, verbose=2)

# evaluate network on the fine-tuning model
print('[INFO] evaluating after fine-tuning...')
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=class_names))

# save model to disk
print('[INFO] save model to disk...')
model.save(args['model'])