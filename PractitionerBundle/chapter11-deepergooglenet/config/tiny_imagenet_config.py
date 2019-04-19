from os import path

# define paths to the training and validation directories
TRAIN_IMAGES = '../datasets/tiny-imagenet-200/train'
VAL_IMAGES = '../datasets/tiny-imagenet-200/val/images'

# define path to file that maps validation filenames to their corresponding class labels
VAL_MAPPINGS = '../datasets/tiny-imagenet-200/val/val_annotations.txt'

# define paths to WordNet hierarchy files which are used to generate our class labels
WORDNET_IDS = '../datasets/tiny-imagenet-200/wnids.txt'
WORD_LABELS = '../datasets/tiny-imagenet-200/words.txt'

# because we do not have testing data, we need to define test size
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define paths to output training, validation and testing HDF5 files
TRAIN_HDF5 = '../datasets/tiny-imagenet-200/hdf5/train.hdf5'
VAL_HDF5 = '../datasets/tiny-imagenet-200/hdf5/val.hdf5'
TEST_HDF5 = '../datasets/tiny-imagenet-200/hdf5/test.hdf5'

# define path to dataset mean
DATASET_MEAN = 'output/tiny-imagenet-200-mean.json'

# define paths to store model, plot, etc.
OUTPUT_PATH = 'output'
MODEL_PATH = path.sep.join([OUTPUT_PATH, 'checkpoints', 'epoch_70.hdf5'])
FIG_PATH = path.sep.join([OUTPUT_PATH, 'deepergooglenet_tinyimagenet.png'])
JSON_PATH = path.sep.join([OUTPUT_PATH, 'deepergooglenet_tinyimagenet.json'])