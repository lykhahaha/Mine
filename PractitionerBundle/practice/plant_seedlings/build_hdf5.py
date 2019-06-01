from imutils import paths
from os import path
import random
import progressbar
import numpy as np
import json
import pickle
import cv2
from sklearn.preprocessing import LabelEncoder
from config import plant_seedlings_config as config
from pyimagesearch.io import HDF5DatasetWriter
from pyimagesearch.preprocessing import AspectAwarePreprocessor

image_paths = list(paths.list_images(config.IMAGES_PATH))
random.shuffle(image_paths)
test_paths = list(paths.list_images(config.TEST_PATH))
image_labels = [p.split(path.sep)[-2] for p in image_paths]

le = LabelEncoder()
image_labels = le.fit_transform(image_labels)
test_labels = [0 for i in range(len(test_paths))]

b_mean, g_mean, r_mean = [], [], []
train_paths, train_labels = image_paths[config.NUM_VAL:], image_labels[config.NUM_VAL:]
val_paths, val_labels = image_paths[:config.NUM_VAL], image_labels[:config.NUM_VAL]

datasets = [
    ('train', train_paths, train_labels, config.TRAIN_HDF5),
    ('val', val_paths, val_labels, config.VAL_HDF5),
    ('test', test_paths, test_labels, config.TEST_HDF5)
]

aap = AspectAwarePreprocessor(256, 256)

for d_type, paths, labels, hdf5_path in datasets:
    widgets = [f'Building {d_type}:', ' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()
    writer = HDF5DatasetWriter(hdf5_path, (len(paths), 256, 256, 3))

    for i, (path, label) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = aap.preprocess(image)

        if d_type == 'train':
            b, g, r = cv2.mean(image)[:3]

            b_mean.append(b)
            g_mean.append(g)
            r_mean.append(r)
        
        writer.add([image], [label])

        pbar.update(i)

    writer.close()
    pbar.finish()

print('Serializing mean of R, G, B...')
mean_rgb = {'R': np.mean(r_mean), 'G': np.mean(g_mean), 'B': np.mean(b_mean)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(mean_rgb))
f.close()

print('Serialzing label mappings...')
f = open(config.LABEL_MAPPINGS, 'wb')
f.write(pickle.dumps(le))
f.close()