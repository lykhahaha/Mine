# USAGE
# python train.py --checkpoints output/checkpoints
# python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_25.hdf5 --start-epoch 25
import matplotlib
matplotlib.use('Agg')

from config import tiny_imagenet_config as config
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.callbacks import EpochCheckpoint, TrainingMonitor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, MeanPreprocessor, SimplePreprocessor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K
import argparse
import json
import sys

# set a high recursion limit so Theano doesn't complain
sys.setrecursionlimit(5000)

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--checkpoints', required=True, help='path to output checkpoint directory')
ap.add_argument('-m', '--model', type=str, help='path to specific model checkpoint to load')
ap.add_argument('-s', '--start-epoch', type=int, default=0, help='epoch to restart training at')
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

# if there is no specific model checkpoint supplied, initialize the network and compile model
if args['model'] is None:
    print('[INFO] compiling model...')
    model = ResNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES, stages=[3, 4, 6], filters=[64, 128, 256, 512], reg=5e-4, dataset='tiny_imagenet')
    opt = SGD(lr=1e-1, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
else:
    print(f"[INFO] loading {args['model']}...")
    model = load_model(args['model'])

    # update learning rate
    print(f'[INFO] old learning rate: {K.get_value(model.optimizer.lr)}')
    K.set_value(model.optimizer.lr, 1e-2)
    print(f'[INFO] new learning rate: {K.get_value(model.optimizer.lr)}')

# construct callbacks
callbacks = [TrainingMonitor(config.FIG_PATH, config.JSON_PATH), EpochCheckpoint(args['checkpoints'], 5, start_at=args['start_epoch'])]

# train the network
model.fit_generator(train_gen.generator(), steps_per_epoch=train_gen.num_images//64, validation_data=val_gen.generator(), validation_steps=val_gen.num_images//64, epochs=50, verbose=2, callbacks=callbacks)

# close dataset
train_gen.close()
val_gen.close()