from os import path
import math

BATCH_SIZE = 64
MAX_EPOCH = 20

INPUT_ROWS = 480
INPUT_COLS = 640

OUTPUT_ROWS = math.ceil(INPUT_ROWS/8)
OUTPUT_COLS = math.ceil(OUTPUT_COLS/8)

TRAIN_IMAGES_PATH = path.sep.join([.., 'salicon', 'images', 'train'])
VAL_PATH = path.sep.join([.., 'salicon', 'images', 'val'])
TEST_PATH = path.sep.join([.., 'salicon', 'images', 'test'])