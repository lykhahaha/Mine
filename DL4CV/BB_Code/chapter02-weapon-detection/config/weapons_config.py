# import the necessary packages
import os

# initialize the base path for the logos dataset
BASE_PATH = "weapons"

# build the path to the annotations and input images
ANNOT_PATH = os.path.sep.join([BASE_PATH, "annotations"])
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])

# define the traing/testing split percentage
TRAIN_SPLIT = 0.75

# build the path to the output training and test .csv files
TRAIN_CSV = os.path.sep.join([BASE_PATH, "retinanet_train.csv"])
TEST_CSV = os.path.sep.join([BASE_PATH, "retinanet_test.csv"])

# build the path to the output classes CSV file
CLASSES_CSV = os.path.sep.join([BASE_PATH, "retinanet_classes.csv"])