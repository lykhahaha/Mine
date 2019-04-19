from os import path

# define base path to where Imagenet dataset (devkit stored on it)
BASE_PATH = path.sep.join(['..', 'datasets', 'imagenet', 'ILSVRC2017'])

# based on base path, derive images base path, image sets path and devkit path
IMAGES_PATH = path.sep.join([BASE_PATH, 'Data', 'CLS-LOC'])
IMAGE_SETS_PATH = path.sep.join([BASE_PATH, 'ImageSets', 'CLS-LOC'])
DEVKIT_PATH = path.sep.join([BASE_PATH, 'devkit', 'data'])

# define path that maps 1000 possible WordNet IDs to class label integers
WORD_IDS = path.sep.join([DEVKIT_PATH, 'map_clsloc.txt'])

# define paths to the training file that maps (partial) image filename to integer class label
TRAIN_LIST = path.sep.join([IMAGE_SETS_PATH, 'train_cls.txt'])

# define paths to validation filenames along with file that contains ground-truth validation labels
VAL_LIST = path.sep.join([IMAGE_SETS_PATH, 'val.txt'])
VAL_LABELS = path.sep.join([DEVKIT_PATH, 'ILSVRC2017_clsloc_validation_ground_truth.txt'])

# define path to the validation files that are blacklisted
VAL_BLACKLIST = path.sep.join([DEVKIT_PATH, 'ILSVRC2017_clsloc_validation_blacklist.txt'])

# since we are not allowed to access testing data, we need to take a number of images from training data and use it instead
NUM_CLASSES = 1000
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define path to the output training, validation and testing lists
MX_OUTPUT = path.sep.join(['..', 'datasets', 'imagenet'])
TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, 'lists', 'train.lst'])
VAL_MX_LIST = path.sep.join([MX_OUTPUT, 'lists', 'val.lst'])
TEST_MX_LIST = path.sep.join([MX_OUTPUT, 'lists', 'test.lst'])

# define path to output training, validation and testing image records
TRAIN_MX_REC = path.sep.join([MX_OUTPUT, 'rec', 'train.rec'])
VAL_MX_REC = path.sep.join([MX_OUTPUT, 'rec', 'val.rec'])
TEST_MX_REC = path.sep.join([MX_OUTPUT, 'rec', 'test.rec'])

# define path to dataset mean
DATASET_MEAN = path.sep.join(['output', 'imagenet_mean.json'])

# define batch size and number of devices used for training
BATCH_SIZE = 128
NUM_DEVICES = 8