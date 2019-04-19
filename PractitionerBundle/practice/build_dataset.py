from config import sr_config as config
from imutils import paths
from pyimagesearch.io import HDF5DatasetWriter
from scipy import misc
import shutil
import os
import cv2

directories = [config.IMAGES, config.LABELS]
for directory in directories:
    if not os.path.isdir(directory):
        os.makedirs(directory)

# load ukbench100
print('[INFO] loading ukbench dataset...')
image_paths = list(paths.list_images(config.DATASET))
total = 0

for i, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    h -= (h%config.SCALE)
    w -= (w%config.SCALE)
    image = image[:h, :w]

    input_image = misc.imresize(image, 1./config.SCALE, interp='bicubic')
    input_image = misc.imresize(input_image, config.SCALE, interp='bicubic')

    for y in range(0, h - config.INPUT_DIM + 1, config.STRIDE):
        for x in range(0, w - config.INPUT_DIM + 1, config.STRIDE):
            crop = input_image[y:y + config.INPUT_DIM, x:x + config.INPUT_DIM]

            target = image[y + config.PAD:y + config.PAD + config.LABEL_SIZE, x + config.PAD:x + config.PAD + config.LABEL_SIZE]

            cv2.imwrite(os.path.sep.join([config.IMAGES, f'{total}.png']), crop)
            cv2.imwrite(os.path.sep.join([config.LABELS, f'{total}.png']), target)

# load images and labels
print('[INFO] loading image and labels...')
image_paths = list(paths.list_images(config.IMAGES))
label_paths = list(paths.list_images(config.LABELS))

# define HDF5 dataset
inputs_hdf5 = HDF5DatasetWriter(config.INPUTS_DB, (len(image_paths), config.INPUT_DIM, config.INPUT_DIM, 3))
outputs_hdf5 = HDF5DatasetWriter(config.OUTPUTS_DB, (len(label_paths), config.LABEL_SIZE, config.LABEL_SIZE, 3))

for image_path, label_path in zip(image_paths, label_paths):
    image = cv2.imread(image_path)
    label = cv2.imread(label_path)

    inputs_hdf5.add([image], [-1])
    outputs_hdf5.add([label], [-1])

# close HDF5 dataset
inputs_hdf5.close()
outputs_hdf5.close()

# remove all images and labels
shutil.rmtree(config.IMAGES)
shutil.rmtree(config.LABELS)