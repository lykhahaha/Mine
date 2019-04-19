import os

# define train and validation paths
TRAIN_IMAGES = '../datasets/tiny-imagenet-200/train'
VAL_IMAGES = '../datasets/tiny-imagenet-200/val/images'

# define class label for validation images
VAL_MAPPING = '../dataset/tiny-imagenet-200/val/val_annotations.txt'

# define WordNet hierarchy files
WORDNET_ID = '../datasets/tiny-imagenet-200/wnids.txt'
WORD_LABELS = '../datasets/tiny-imagenet-200/words.txt'

# define test size
NUM_CLASSES = 20
NUM_TEST_SIZE = 500 * NUM_CLASSES

# define paths to hdf5 files
TRAIN_HDF5 = '../datasets/tiny-imagenet-200/hdf5/train.hdf5'
VAL_HDF5 = '../datasets/tiny-imagenet-200/hdf5/val.hdf5'
TEST_HDF5 = '../datasets/tiny-imagenet-200/hdf5/test.hdf5'

# define output path, checkpoint and dataset mean
OUTPUT_PATH = 'output'
EPOCH = 70
DATASET_MEAN = os.path.sep.join([OUTPUT_PATH, 'tiny_imagenet_200_mean.json'])
FIG_PATH = os.path.sep.join([OUTPUT_PATH, 'deepergooglenet_tinyimagenet.png'])
JSON_PATH = os.path.sep.join([OUTPUT_PATH, 'deepergooglenet_tinyimagenet.json'])
MODEL_PATH = os.path.sep.join([OUTPUT_PATH, 'checkpoints', f'epoch_{EPOCH}.hdf5'])