# USAGE
# python train_alexnet.py
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from config import dogs_and_cats_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor, PatchPreprocessor, MeanPreprocessor
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
import json
import os

# construct training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, zoom_range=0.15, horizontal_flip=True)

# load RGB mean for training set
means = json.loads(open(config.DATASET_MEAN, 'r').read())

# initialize image preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

# initialize training and validation dataset generators
print('[INFO] loading dataset...')
train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5, batch_size=64*config.G, preprocessors=[pp, mp, iap], aug=aug, classes=2)
val_gen = HDF5DatasetGenerator(config.VAL_HDF5, batch_size=64*config.G, preprocessors=[sp, mp, iap], classes=2)

# initialize the optimizer
if config.G <= 1:
    print(f'[INFO] compiling model with 1 GPU...')
    model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
else:
    print(f'[INFO] compiling model with {config.G} GPU...')
    # opt = Adam(lr=1e-3)
    with tf.device('/cpu:0'):
        model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
    model = multi_gpu_model(model, gpus=config.G)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# construct set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, f'{os.getpid()}.png'])
callbacks = [TrainingMonitor(path)]

# train the network
print('[INFO] training model...')
model.fit_generator(train_gen.generator(), steps_per_epoch=train_gen.num_images//(64*config.G), validation_data=val_gen.generator(), validation_steps=val_gen.num_images//(64*config.G), epochs=75, callbacks=callbacks, verbose=2)

# save model to file
print('[INFO] serializing model...')
model.save(config.MODEL_PATH, overwrite=True)

# close HDF5 datasets
train_gen.close()
val_gen.close()