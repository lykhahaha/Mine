# USAGE
# python train_decay.py --model output/resnet_tinyimagenet_decay.hdf5 --output output
import matplotlib
matplotlib.use('Agg')

from config import tiny_imagenet_config as config
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.preprocessing import SimplePreprocessor, MeanPreprocessor, ImageToArrayPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import argparse
import json
import sys
import os

# set a high recursion limit so Theano doesn't complain
sys.setrecursionlimit(5000)

# define total number of epochs to train along with initial learning rate
MAX_EPOCH = 75
INIT_LR = 1e-1

def poly_decay(epoch):
    # define total number of epochs, base learning rate and power of polynomial
    max_epoch = MAX_EPOCH
    base_lr = INIT_LR
    power = 1.0

    # compute new learning rate based on polynomial decay
    alpha = base_lr * (1 - epoch / float(max_epoch))**power

    return alpha

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='path to output model')
ap.add_argument('-o', '--output', required=True, help='path to output directory (plot, logs)')
args = vars(ap.parse_args())

# construct argument parser nad parse the argument
aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, zoom_range=0.15, horizontal_flip=True)

# load RGB means for training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize image preprocessors
sp, mp, iap = SimplePreprocessor(64, 64), MeanPreprocessor(means['R'], means['G'], means['B']), ImageToArrayPreprocessor()

# initialize training and validation dataset generators
train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, preprocessors=[sp, mp, iap], aug=aug, classes=config.NUM_CLASSES)
val_gen = HDF5DatasetGenerator(config.VAL_HDF5, 64, preprocessors=[sp, mp, iap], aug=aug, classes=config.NUM_CLASSES)

# construct callbacks
fig_path = os.path.sep.join([args['output'], f'{os.getpid()}.png'])
json_path = os.path.sep.join([args['output'], f'{os.getpid()}.json'])
callbacks = [TrainingMonitor(fig_path, json_path=json_path), LearningRateScheduler(poly_decay)]

# initialize and compile model
print('[INFO] compiling model...')
model = ResNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES, stages=[3, 4, 6], filters=[64, 128, 256, 512], reg=5e-4, dataset='tiny_imagenet')
opt = SGD(lr=1e-1, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
model.fit_generator(train_gen.generator(), steps_per_epoch=train_gen.num_images//64, validation_data=val_gen.generator(), validation_steps=val_gen.num_images//64, epochs=MAX_EPOCH, verbose=2, callbacks=callbacks)

# save model to disk
print(('[INFO] serializing model...'))
model.save(args['model'])

# close dataset
train_gen.close()
val_gen.close()