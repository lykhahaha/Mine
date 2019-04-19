# USAGE
# python augmentation_video.py --image jemma.png --output output
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import argparse

# Construct argument parse and parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-o', '--output', required=True, help='path to putput directory to store augmentation examples')
ap.add_argument('-p', '--prefix', type=str, default='image', help='output filename prefix')
args = vars(ap.parse_args())

# load input image, convert it to numpy array, reshape to an extra dimension
print('[INFO] loading example image...')
image = load_img(args['image'])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# construct image generator for data augmentation the initialize total number of image generated
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
total=0

# construct actual Python generator
print('[INFO] generating images...')
image_gen = aug.flow(image, batch_size=1, save_to_dir=args['output'], save_prefix=args['prefix'], save_format='png')

# loop over examples from our image data augmentation generator
for image in image_gen:
    # increase our counter
    total+=1
    # if reached 10 example, break
    if total == 10:
        break