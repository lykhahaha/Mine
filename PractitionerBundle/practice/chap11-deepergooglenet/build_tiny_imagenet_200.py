from config import tiny_imagenet_config as config
from pyimagesearch.io import HDF5DatasetWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import cv2
import progressbar
import os
import json

# grab the list of images and extract labels from them
train_paths = list(paths.list_images(config.TRAIN_IMAGES))
train_labels = [p.split(os.path.sep)[-3] for p in train_paths]
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)

# obtain validation images and labels
f = open(config.VAL_MAPPING).read().strip().split('\n')
val_map = [p.split('\t')[:2] for p in f]
val_paths = [os.path.sep.join(config.VAL_IMAGES, p[0]) for p in val_map]
val_labels = le.transform([p[1] for p in val_map])

# partition data
train_paths, test_paths, train_labels, test_labels = train_test_split(train_paths, train_labels, test_size=config.NUM_TEST_SIZE, stratify=train_labels, random_state=42)

# construct means of R, G, B
R, G, B = [], [], []

# construct DATASET
DATASET = [
    ('train', train_paths, train_labels, config.TRAIN_HDF5),
    ('val', val_paths, val_labels, config.VAL_HDF5),
    ('test', test_paths, test_labels, config.TEST_HDF5)
]

# loop over DATASeT
for d_type, image_paths, labels, output_path in DATASET:
    # construct hdf5 dataset writer
    writer = HDF5DatasetWriter((len(labels), 64, 64, 3), output_path)
    writer.storeClassLabels(le.classes_)

    # construct progress bar
    widgets = [f'Building {d_type}: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(labels), widgets=widgets).start()

    # loop over each image
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        image = cv2.imread(image_path)

        if d_type == 'train':
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        
        writer.add([image], [label])
        pbar.update(i)
    
    # close dataset
    writer.close()
    pbar.finish()

# store means to disk
print('[INFO] serializing means...')
means = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(means))
f.close()