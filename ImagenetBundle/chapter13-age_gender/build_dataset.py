from config import age_gender_deploy as deploy
from config import age_gender_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.utils import AgeGenderHelper
from pyimagesearch.io import HDF5DatasetWriter
from pyimagesearch.preprocessing import AspectAwarePreprocessor, MeanPreprocessor
from pyimagesearch.nn.conv import SaliencyNet
import numpy as np
import progressbar
import pickle
import json
import cv2
import os
from keras import backend as K

def build_hdf5(dataset, dataset_mean_path, label_encoder_path):
    # list of R, G, B means
    R, G, B = [], [], []

    # initialize image preprocessor
    aap = AspectAwarePreprocessor(256, 256)

    # loop over DATASETS
    for d_type, paths, labels, output_path in dataset:
        # construct HDF% dataset writer
        writer = HDF5DatasetWriter((len(labels), 256, 256, 3), output_path)
        # construct progress bar
        widgets = [f'Building {d_type}: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(labels), widgets=widgets).start()

        for i, (path, label) in enumerate(zip(paths, labels)):
            image = cv2.imread(path)
            
            image = aap.preprocess(image)

            if d_type == 'train':
                b, g, r = cv2.mean(image)[:3]
                R.append(r)
                G.append(g)
                B.append(b)
            
            writer.add([image], [label])
            pbar.update(i)

        writer.close()
        pbar.finish()

    if not os.path.exists(config.OUTPUT_BASE):
        os.makedirs(config.OUTPUT_BASE)

    # serialize means of R, G, B
    print('[INFO] serialzing means...')
    D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
    f = open(dataset_mean_path, 'w')
    f.write(json.dumps(D))
    f.close()

    # serialize label encoder
    print('[INFO] serializing label encoder...')
    f = open(label_encoder_path, 'wb')
    f.write(pickle.dumps(le))
    f.close()

# initialize helper class
agh = AgeGenderHelper(config, deploy)

if config.DATASET == 'IOG':
    # build set of image paths and class labels
    print('[INFO] building paths and labels...')
    train_paths, train_labels, test_paths, test_labels = agh.build_paths_and_labels_iog_preprocessed()

    # define number of validation and testing size
    num_val = int(len(train_labels) * config.NUM_VAL_IMAGES)

    # our class labels are represented as strings, so encode them
    print(f'[INFO] encoding labels {config.SALIENCY_INFO}...')
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

    # partition training set into training and validation test
    print('[INFO] constructing validation data...')
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=num_val, stratify=train_labels, random_state=42)

    # define DATASET
    DATASETS = [
        ('train', train_paths, train_labels, config.TRAIN_HDF5),
        ('val', val_paths, val_labels, config.VAL_HDF5),
        ('test', test_paths, test_labels, config.TEST_HDF5)
    ]

    build_hdf5(DATASETS, config.DATASET_MEAN, config.LABEL_ENCODER_PATH)

elif config.DATASET == 'ADIENCE':
    # loop over each cross validation of image paths and class labels
    for i, (train_paths, train_labels, test_paths, test_labels, test_paths_frontal, test_labels_frontal) in enumerate(agh.build_paths_and_labels_adience()):
        # define number of validation and testing size
        num_val = int(len(train_labels) * config.NUM_VAL_IMAGES)

        # our class labels are represented as strings, so encode them
        print(f'[INFO] encoding labels for cross-validation {i}...')
        le = LabelEncoder()
        train_labels = le.fit_transform(train_labels)
        test_labels = le.transform(test_labels)
        test_labels_frontal = le.transform(test_labels_frontal)

        # partition training set into training and validation test
        print(f'[INFO] constructing validation data for cross-validation {i}...')
        train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=num_val, stratify=train_labels, random_state=42)

        # define DATASET
        DATASETS = [
            ('train', train_paths, train_labels, config.TRAIN_HDF5S[i]),
            ('val', val_paths, val_labels, config.VAL_HDF5S[i]),
            ('test', test_paths, test_labels, config.TEST_HDF5S[i]),
            ('frontalized test', test_paths_frontal, test_labels_frontal, config.TEST_HDF5S_FRONTAL[i])
        ]

        build_hdf5(DATASETS, config.DATASET_MEANS[i], config.LABEL_ENCODER_PATHS[i])