# USAGE
# For adam optimizer, start at 1e-3 (default) after 40 epochs decrease to 1e-4 after 20 epochs decrease to 1e-5 for more 20 epochs. Change # of epochs at model.fit_generator
# python train.py --checkpoints output/checkpoints
# python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_40.hdf5 --start-epoch 40
# python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_60.hdf5 --start-epoch 60

# For SGD optimizer, start at 1e-2 (default) after 35 epochs decrease to 1e-3 after 10 epochs decrease to 1e-4 for more 30 epochs. Change # of epochs at model.fit_generator
# python train.py --checkpoints output/checkpoints
# python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_35.hdf5 --start-epoch 35
# python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_45.hdf5 --start-epoch 45

import matplotlib
matplotlib.use('Agg')

from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor, MeanPreprocessor
from pyimagesearch.callbacks import EpochCheckpoint, TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import DeeperGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
import argparse
import json

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--checkpoints', required=True, help='path to output checkpoint directory')
ap.add_argument('-m', '--model', type=str, help='path to specific model checkpoint to load')
ap.add_argument('-s', '--start-epoch', type=int, default=0, help='epoch to restart training at')
args = vars(ap.parse_args())

# construct training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=18, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, zoom_range=0.15, horizontal_flip=True)

# load RGB means for training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize image preprocessors
sp, mp, iap = SimplePreprocessor(64, 64), MeanPreprocessor(means['R'], means['G'], means['B']), ImageToArrayPreprocessor()

# initialize training and validation dataset generators
train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, preprocessors=[sp, mp, iap], aug=aug, classes=config.NUM_CLASSES)
val_gen = HDF5DatasetGenerator(config.VAL_HDF5, 64, preprocessors=[sp, mp, iap], aug=aug, classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, initialize network and compile model
if args['model'] is None:
    print('[INFO] compiling the network...')
    model = DeeperGoogLeNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES, reg=2e-4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# otherwise load the checkpoint from disk
else:
    print(f"[INFO] loading {args['model']}")
    model = load_model(args['model'])

    # update learning rate
    print(f'[INFO] old learning rate: {K.get_value(model.optimizer.lr)}')
    K.set_value(model.optimizer.lr, 1e-5)
    print(f'[INFO] new learning rate: {K.get_value(model.optimizer.lr)}')

# construct set of callbacks
callbacks = [
    EpochCheckpoint(args['checkpoints'], every=5, start_at=args['start_epoch']),
    TrainingMonitor(config.FIG_PATH, json_path=config.JSON_PATH, start_at=args['start_epoch'])
]

# train the network
model.fit_generator(train_gen.generator(), steps_per_epoch=train_gen.num_images//64, validation_data=val_gen.generator(), validation_steps=val_gen.num_images//64, epochs=40, verbose=2, callbacks=callbacks)

# close the database
train_gen.close()
val_gen.close()