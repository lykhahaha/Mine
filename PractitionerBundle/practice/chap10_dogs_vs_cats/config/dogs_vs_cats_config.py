# define images path
IMAGES_PATH = '../datasets/kaggle_dogs_vs_cats/train'

# define size of test and validation to partition images
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# define path to training, validation and test HDF5
TRAIN_HDF5 = '../datasets/kaggle_dogs_vs_cats/hdf5/train.hdf5'
VAL_HDF5 = '../datasets/kaggle_dogs_vs_cats/hdf5/val.hdf5'
TEST_HDF5 = '../datasets/kaggle_dogs_vs_cats/hdf5/test.hdf5'

# define path to output model
MODEL_PATH = 'output/alexnet_dogs_vs_cats.model'

# define path to mean of R, G, B channels of images
DATASET_MEAN = 'output/dogs_vs_cats_mean.json'

# define directory to store loss/accuracy plot
OUTPUT_PATH = 'output'