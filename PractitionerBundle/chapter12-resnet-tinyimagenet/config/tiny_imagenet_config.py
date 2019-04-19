import os

# define training and validation paths
TRAIN_IMAGES = '../datasets/tiny-imagenet-200/train'
VAL_IMAGES = '../dataset/tiny-imagenet-200/val/images'

# define WordNet hierarchy files
WORDNET_IDS = '../dataset/tiny-imagenet-200/wnids.txt'
WORD_LABELS = '../dataset/tiny-imagenet-200/words.txt'

# define validation mapping for initializing validation images and labels
VAL_MAPPINGs = '../datasets/tin-imagenet-200/val/val_annotations.txt'

# define test size
NUM_CLASSES = 200
NUM_TEST_SIZE = 50 * NUM_CLASSES

# define hdf5 files for training, validation and testing
TRAIN_HDF5 = '../datasets/tiny-imagenet-200/hdf5/train.hdf5'
VAL_HDF5 = '../datasets/tiny-imagenet-200/hdf5/val.hdf5'
TEST_HDF5 = '../datasets/tiny-imagenet-200/hdf5/test.hdf5'

# define output path, output directory, checkpoints nad dataset mean
OUTPUT_PATH = 'output'
DATASET_MEAN = os.path.sep.join([OUTPUT_PATH, 'tiny-imagenet-200-mean.json'])
MODEL_PATH = os.path.sep.join([OUTPUT_PATH, 'resnet_tinyimagenet.hdf5'])
FIG_PATH = os.path.sep.join([OUTPUT_PATH, 'resnet56_tinyimagenet.png'])
JSON_PATH = os.path.sep.join([OUTPUT_PATH, 'resnet56_tinyimagenet.json'])