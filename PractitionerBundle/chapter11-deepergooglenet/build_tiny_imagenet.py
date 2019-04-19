from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# grab the list of training images and extract them to get labels
train_paths = list(paths.list_images(config.TRAIN_IMAGES))
train_labels = [image_path.split(os.path.sep)[-3] for image_path in train_paths]

# convert label to integers
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)

# partition training data by using stratifying
train_paths, test_paths, train_labels, test_labels = train_test_split(train_paths, train_labels, test_size=config.NUM_TEST_IMAGES, stratify=train_labels, random_state=42)

# load validation filename and maps label to image
M = open(config.VAL_MAPPINGS).read().strip().split('\n') # strip: remove \n in the end
M = [p.split('\t')[:2] for p in M]
val_paths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
val_labels = le.transform([m[1] for m in M])

# construct a list of pairing training, validation and testing image paths along with their corresponding labels and output HDF5 files
DATASET = [
    ('train', train_paths, train_labels, config.TRAIN_HDF5),
    ('val', val_paths, val_labels, config.VAL_HDF5),
    ('test', test_paths, test_labels, config.TEST_HDF5)
]

# initialize list of RGB channels
R, G, B = [], [], []

# loop over dataset tuples
for d_type, image_paths, labels, output_path in DATASET:
    # create HDF5 file
    print(f'[INFO] building {d_type}...')
    writer = HDF5DatasetWriter((len(labels), 64, 64, 3), output_path)

    # initialize progress bar
    widgets = ['Building dataset: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(labels), widgets=widgets).start()

    # loop over image path
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        image = cv2.imread(image_path)

        if d_type == 'train':
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add([image], [label])
        pbar.update(i)

    # close the dataset
    writer.close()
    pbar.finish()

# construct a dictionary of average then save JSON file
print('[INFO] serializing mean...')
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(D))
f.close()