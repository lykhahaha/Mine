from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import argparse

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image generated')
ap.add_argument('-o', '--output', required=True, help='path to the output directory')
ap.add_argument('-p', '--prefix', type=str, default='image', help='prefix of name of augmented images')
args = vars(ap.parse_args())

# load the image, convert it to Keras-compatible array, add extra dimension to it
print('[INFO] loading the image...')
image = load_img(args['image'])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# construct data augmentation and counter
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
total = 0

# construct Python generator
print('[INFO] generating images...')
image_gen = aug.flow(image, batch_size=1, save_to_dir=args['output'], save_prefix=args['prefix'], save_format='jpg')

# save image to disk
for image in image_gen:
    # increase the counter
    total += 1

    # if reaching 10 generated images, stop
    if total == 10:
        break