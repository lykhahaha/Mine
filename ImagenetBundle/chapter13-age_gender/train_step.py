# USAGE
# python train.py --checkpoints checkpoints\age
# python train.py --checkpoints checkpoints\age --model checkpoints\age\epoch_10.hdf5 --start-epoch 10
from config import age_gender_deploy as deploy
from config import age_gender_config as config
from pyimagesearch.nn.conv import AgeGenderNet
from pyimagesearch.utils import AgeGenderHelper
from pyimagesearch.preprocessing import SimplePreprocessor, MeanPreprocessor, PatchPreprocessor, ImageToArrayPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.callbacks import EpochCheckpoint, TrainingMonitor, OneOffAccuracy, ModelCheckpointsAdvanced
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K
from keras.callbacks import LearningRateScheduler
import argparse
import numpy as np
import pickle
import json
import os

# define total number of epochs to train along with initial learning rate
MAX_EPOCH = 150
INIT_LR = 1e-1
POWER = 1.0
IMAGE_SIZE = 227

def decay(epoch):
    epoch += args['start_epoch']
    alpha = INIT_LR * (1 - (epoch/float(MAX_EPOCH))) ** POWER

    return alpha

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
# ap.add_argument('-c', '--checkpoints', required=True, help='path to output checkpoint directory')
# ap.add_argument('-m', '--model', type=str, help='path to specific model checkpoint to load')
ap.add_argument('-s', '--start-epoch', type=int, default=0, help='epoch to restart at')
args = vars(ap.parse_args())

def training(aug, means_path, train_hdf5_path, val_hdf5_path, fig_path, json_path, label_encoder_path, best_weight_path, checkpoint_path, cross_val=None):
    # load RGB means
    means = json.loads(open(means_path).read())

    # initialize image preprocessors
    sp, mp, pp, iap = SimplePreprocessor(IMAGE_SIZE, IMAGE_SIZE), MeanPreprocessor(means['R'], means['G'], means['B']), PatchPreprocessor(IMAGE_SIZE, IMAGE_SIZE), ImageToArrayPreprocessor()

    # initialize training and validation image generator
    train_gen = HDF5DatasetGenerator(train_hdf5_path, config.BATCH_SIZE, preprocessors=[pp, mp, iap], aug=aug, classes=config.NUM_CLASSES)
    val_gen = HDF5DatasetGenerator(val_hdf5_path, config.BATCH_SIZE, preprocessors=[sp, mp, iap], aug=aug, classes=config.NUM_CLASSES)

    metrics = ['accuracy']
    if config.DATASET_TYPE == 'age':
        le = pickle.loads(open(label_encoder_path, 'rb').read())
        agh = AgeGenderHelper(config, deploy)
        one_off_mappings = agh.build_oneoff_mappings(le)

        one_off = OneOffAccuracy(one_off_mappings)
        metrics.append(one_off.one_off_accuracy)

    # construct callbacks
    callbacks = [TrainingMonitor(fig_path, json_path=json_path, start_at=args['start_epoch']), EpochCheckpoint(checkpoint_path, every=5, start_at=args['start_epoch']), ModelCheckpointsAdvanced(best_weight_path, json_path=json_path, start_at=args['start_epoch'])]

    if cross_val is None:
        print('[INFO] compiling model...')
    else:
        print(f'[INFO] compiling model for cross validation {cross_val}...')
    
    if args['start_epoch'] == 0:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        model = AgeGenderNet.build(IMAGE_SIZE, IMAGE_SIZE, 3, config.NUM_CLASSES, reg=5e-4)
        opt = SGD(lr=INIT_LR, momentum=0.9, decay=5e-4)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)
    else:
        model_path = os.path.sep.join([checkpoint_path, f"epoch_{args['start_epoch']}.hdf5"])
        print(f"[INFO] loading {model_path}...")
        if config.DATASET_TYPE == 'age':
            model = load_model(model_path, custom_objects={'one_off_accuracy': one_off.one_off_accuracy})
        elif config.DATASET_TYPE == 'gender':
            model = load_model(model_path)

        # update learning rate
        print(f'[INFO] old learning rate: {K.get_value(model.optimizer.lr)}')
        K.set_value(model.optimizer.lr, INIT_LR)
        print(f'[INFO] new learning rate: {K.get_value(model.optimizer.lr)}')

    # train the network
    if cross_val is None:
        print('[INFO] training the network...')
    else:
        print(f'[INFO] training the network for cross validation {cross_val}...')
    model.fit_generator(train_gen.generator(), steps_per_epoch=train_gen.num_images//config.BATCH_SIZE, validation_data=val_gen.generator(), validation_steps=val_gen.num_images//config.BATCH_SIZE, epochs=MAX_EPOCH-args['start_epoch'], verbose=2, callbacks=callbacks)

    # close dataset
    train_gen.close()
    val_gen.close()

# construct data iterator for image augmentation
aug = ImageDataGenerator(rotation_range=15, width_shift_range=0.15, height_shift_range=0.15, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)

if config.DATASET == 'IOG':
    training(aug, config.DATASET_MEAN, config.TRAIN_HDF5, config.VAL_HDF5, config.FIG_PATH, config.JSON_PATH, config.LABEL_ENCODER_PATH, config.BEST_WEIGHT, config.CHECKPOINT)

elif config.DATASET == 'ADIENCE':
    for i in range(config.NUM_FOLD_PATHS):
        training(aug, config.DATASET_MEANS[i], config.TRAIN_HDF5S[i], config.VAL_HDF5S[i], config.FIG_PATHS[i], config.JSON_PATHS[i], config.LABEL_ENCODER_PATHS[i], config.BEST_WEIGHTS[i], config.CHECKPOINTS[i], i)