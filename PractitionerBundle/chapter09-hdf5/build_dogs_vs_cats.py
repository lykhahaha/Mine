# USAGE
# python build_dogs_vs_cats.py
from config import dogs_and_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

os.chdir('F:/')
print(os.getcwd())
# grab the paths to the images
train_paths = list(paths.list_images(config.IMAGES_PATH))
train_labels = [p.split(os.path.sep)[-1].split('.')[0] for p in train_paths]
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)

# perform stratified sampling from the training set to build the testing split from the training data
train_paths, test_paths, train_labels, test_labels = train_test_split(train_paths, train_labels, test_size=config.NUM_TEST_IMAGES, stratify=train_labels, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=config.NUM_VAL_IMAGES, stratify=train_labels, random_state=42)

# construct a list pairing the training, validation and testing image paths along with their corresponding labels and output HDF5 files
datasets = [
    ('train', train_paths, train_labels, config.TRAIN_HDF5),
    ('val', val_paths, val_labels, config.VAL_HDF5),
    ('test', test_paths, test_labels, config.TEST_HDF5)
]

# initialize the image preprocessor and list of RGB channel averages
aap = AspectAwarePreprocessor(256, 256)
R, G, B = [], [], []

# loop over the dataset tuples
for d_type, paths, labels, output_path in datasets:
    # create HDF5 writer
    print(f'[INFO] building {output_path}...')
    writer = HDF5DatasetWriter((len(labels), 256, 256, 3), output_path)

    # initialize progress bar
    widgets = ['Building dataset: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(labels), widgets=widgets).start()

    # loop over the image paths
    for i, (path, label) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = aap.preprocess(image)

        # if we are building the training dataset, then compute the mean of each channel in the image, then update the respective lists
        if d_type == 'train':
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add the image and label to the HDF5 dataset
        writer.add([image], [label]) # data.extend(np.ones((2, 256, 256, 3))), data[0].shape -> (256, 256, 3), but data.extend(np.ones((256, 256, 3))), data[0].shape -> (256, 3)
        pbar.update(i)

    # close the HDF5 writer
    pbar.finish()
    writer.close()

# construct a dictionary of average, then save means to a JSON file
print('[INFO] serializing means...')
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(D))
f.close()