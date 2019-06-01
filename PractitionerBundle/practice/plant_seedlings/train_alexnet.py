from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.preprocessing import SimplePreprocessor, MeanPreprocessor, PatchPreprocessor
from config import plant_seedlings_config as config
from pyimagesearch.nn.conv import AlexNet
from keras.preprocessing.image import ImageDataGenerator
import json
import pickle

mean = json.loads(open(config.DATASET_MEAN).read())
le = pickle.loads(open(config.LABEL_MAPPINGS, 'rb').read())

sp, mp, pp = SimplePreprocessor(224, 224), MeanPreprocessor(mean['R'], mean['G'], mean['B']), PatchPreprocessor(224, 224)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, horizontal_flip=True)

train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5, preprocessors=[pp, mp], aug=aug, batch_size=64, num_classes=len(le.classes_))
val_gen = HDF5DatasetGenerator(config.VAL_HDF5, preprocessors=[sp, mp], aug=aug, batch_size=64, num_classes=len(le.classes_))

model = AlexNet.build(224, 224, 3, len(le.classes_))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen.generator(), steps_per_epoch=train_gen.num_images//64, epochs=100, verbose=2, )