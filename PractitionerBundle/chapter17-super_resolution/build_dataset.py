from pyimagesearch.io import HDF5DatasetWriter
from config import sr_config as config
from imutils import paths
from scipy import misc
import shutil
import random
import cv2
import os

# if output directories do not exist, create them
for p in [config.IMAGES, config.LABELS]:
    if not os.path.exists(p):
        os.makedirs(p)

# grab image paths and initialize total number of crops processed
print('[INFO] creating temp images...')
image_paths = list(paths.list_images(config.INPUT_IMAGES))
random.shuffle(image_paths)
total = 0

# loop over image paths
for image_path in image_paths:
    image = cv2.imread(image_path)

    # get dimensions of input image amd crop image such that it tiles nicely when we generate training data + labels
    h, w = image.shape[:2]
    w -= int(w % config.SCALE)
    h -= int(h % config.SCALE)
    image = image[:h, :w]

    # to generate our training images, we first need to downscale image by scale factor...then upscale it back to original size
    # this process allows us to generate low resolution inputs that we'll then learn to reconstruct high resolution versions from
    scaled = misc.imresize(image, 1./config.SCALE, interp='bicubic')
    scaled = misc.imresize(scaled, config.SCALE, interp='bicubic')

    # slide window from left-to-right and top-to-bottom
    for y in range(0, h - config.INPUT_DIM + 1, config.STRIDE):
        for x in range(0, w - config.INPUT_DIM + 1, config.STRIDE):
            # this ROI will serve as input to network
            crop = scaled[y:y + config.INPUT_DIM, x:x + config.INPUT_DIM]

            # this ROI will be target output from network
            target = image[y + config.PAD:y + config.PAD + config.LABEL_SIZE, x + config.PAD:x + config.PAD + config.LABEL_SIZE]

            # construct crop and target output image paths
            crop_path = os.path.sep.join([config.IMAGES, f'{total}.png'])
            target_path = os.path.sep.join([config.LABELS, f'{total}.png'])

            # write images to disk
            cv2.imwrite(crop_path, crop)
            cv2.imwrite(target_path, target)

            total += 1

# grab paths to the images
print('[INFO] building HDF5 datasets...')
input_paths = sorted(list(paths.list_images(config.IMAGES)))
output_paths = sorted(list(paths.list_images(config.LABELS)))

# initialize HDF5 datasets
input_writer = HDF5DatasetWriter((len(input_paths), config.INPUT_DIM, config.INPUT_DIM, 3), config.INPUT_DB)
output_writer = HDF5DatasetWriter((len(output_paths), config.LABEL_SIZE, config.LABEL_SIZE, 3), config.OUTPUT_DB)

# loop over images
for input_path, output_path in zip(input_paths, output_paths):
    # load 2 images and add them to datasets
    input_image, output_image = cv2.imread(input_path), cv2.imread(output_path)
    input_writer.add([input_image], [-1])
    output_writer.add([output_image], [-1])

# close HDF5 dataset
input_writer.close()
output_writer.close()

# delete temporary output directories
print('[INFO] cleaning up...')
shutil.rmtree(config.IMAGES)
shutil.rmtree(config.LABELS)