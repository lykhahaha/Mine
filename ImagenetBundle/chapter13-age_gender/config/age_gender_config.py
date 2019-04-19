from os import path
import os

# define dataset to use
DATASET = 'IOG' # ADIENCE or IOG

# define type of dataset we are training
DATASET_TYPE = 'age' # age or gender
NUM_FOLD_PATHS = 5

# define batch size
BATCH_SIZE = 64
# define percentage of validation and testing images relative to number of training images
NUM_VAL_IMAGES = 0.1
IMAGE_SIZE = 227

if DATASET == 'IOG':
    # define base paths to the faces dataset and output path
    BASE_PATH = path.sep.join(['..', 'datasets', 'IoG'])

    OUTPUT_BASE = 'output_iog'

    MAT_TEST_PATH = path.sep.join([BASE_PATH, 'AgeGenderClassification', 'eventest.mat'])
    BEST_WEIGHT = path.sep.join([OUTPUT_BASE, f'{DATASET_TYPE}_best_weights.hdf5'])
    WRONG_BASE = 'wrong_iog'

    # check to see if we are working with 'age portion of dataset
    if DATASET_TYPE == 'age':
        # define number of labels for 'age' dataset
        NUM_CLASSES = 7
        # initialize age and gender paths for construct age and gender label of each face
        DATASET_PATH = os.path.sep.join([BASE_PATH, 'age'])
        DATASET_TEST_PATH = os.path.sep.join([BASE_PATH, 'age_test'])

    elif DATASET_TYPE == 'gender':
        # define number of labels for 'gender' dataset
        NUM_CLASSES = 2
        # initialize age and gender paths for construct age and gender label of each face
        DATASET_PATH = os.path.sep.join([BASE_PATH, 'gender'])
        DATASET_TEST_PATH = os.path.sep.join([BASE_PATH, 'gender_test'])

    # define paths of json and figure for training
    FIG_PATH = os.path.sep.join([OUTPUT_BASE, f'{DATASET_TYPE}_figure.png']) 
    JSON_PATH = os.path.sep.join([OUTPUT_BASE, f'{DATASET_TYPE}_monitor.json'])

    # define path to output training, validation and testing hdf5
    TRAIN_HDF5 = path.sep.join([BASE_PATH, 'hdf5', f'{DATASET_TYPE}_train.hdf5'])
    VAL_HDF5 = path.sep.join([BASE_PATH, 'hdf5', f'{DATASET_TYPE}_val.hdf5'])
    TEST_HDF5 = path.sep.join([BASE_PATH, 'hdf5', f'{DATASET_TYPE}_test.hdf5'])

    # Label Encoder
    LABEL_ENCODER_PATH = path.sep.join([OUTPUT_BASE, f'{DATASET_TYPE}_le.cpickle'])

    # define path to dataset mean
    DATASET_MEAN = path.sep.join([OUTPUT_BASE, f'{DATASET_TYPE}_mean.json'])

    CHECKPOINT = os.path.sep.join(['checkpoints', 'IoG', DATASET_TYPE])

elif DATASET == 'ADIENCE':
    # define base paths to the faces dataset and output path
    BASE_PATH = path.sep.join(['..', 'datasets', 'adience'])

    OUTPUT_BASE = 'output'
    OUTPUT_BASE_FRONTAL = 'output_frontal'
    WRONG_BASE = 'wrong'
    WRONG_BASE_FRONTAL = 'wrong_frontal'

    # based on base path, derive images path and folds path
    IMAGES_PATH = path.sep.join([BASE_PATH, 'aligned'])
    LABELS_PATH = path.sep.join([BASE_PATH, 'folds'])

    FIG_PATHS, JSON_PATHS = [], []
    TRAIN_HDF5S, VAL_HDF5S, TEST_HDF5S, TEST_HDF5S_FRONTAL = [], [], [], []
    LABEL_ENCODER_PATHS, DATASET_MEANS, BEST_WEIGHTS = [], [], []
    CHECKPOINTS = []

    for i in range(NUM_FOLD_PATHS):
        # define paths of json and figure for training
        FIG_PATHS.append(os.path.sep.join([OUTPUT_BASE, f'{DATASET_TYPE}_figure_{i}.png']) )
        JSON_PATHS.append(os.path.sep.join([OUTPUT_BASE, f'{DATASET_TYPE}_monitor_{i}.json']))

        # define path to output training, validation and testing hdf5
        TRAIN_HDF5S.append(path.sep.join([BASE_PATH, 'hdf5', f'{DATASET_TYPE}_train_{i}.hdf5']))
        VAL_HDF5S.append(path.sep.join([BASE_PATH, 'hdf5', f'{DATASET_TYPE}_val_{i}.hdf5']))
        TEST_HDF5S.append(path.sep.join([BASE_PATH, 'hdf5', f'{DATASET_TYPE}_test_{i}.hdf5']))
        TEST_HDF5S_FRONTAL.append(path.sep.join([BASE_PATH, 'hdf5', f'{DATASET_TYPE}_test_{i}_frontal.hdf5']))

        BEST_WEIGHTS.append(path.sep.join([OUTPUT_BASE, f'{DATASET_TYPE}_best_weights_{i}.hdf5']))

        # Label Encoder
        LABEL_ENCODER_PATHS.append(path.sep.join([OUTPUT_BASE, f'{DATASET_TYPE}_le_{i}.cpickle']))

        # define path to dataset mean
        DATASET_MEANS.append(path.sep.join([OUTPUT_BASE, f'{DATASET_TYPE}_mean_{i}.json']))

        CHECKPOINTS.append(os.path.sep.join(['checkpoints', 'adience', f'cross{i}', DATASET_TYPE]))


    # check to see if we are working with 'age portion of dataset
    if DATASET_TYPE == 'age':
        # define number of labels for 'age' dataset
        NUM_CLASSES = 8

    elif DATASET_TYPE == 'gender':
        # define number of labels for 'gender' dataset
        NUM_CLASSES = 2

else:
    raise ValueError('Dataset name should be ADIENCE or IOG')