from config import sr_config as config
from keras.models import load_model
from scipy import misc
import numpy as np
import argparse
import cv2

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-b', '--baseline', required=True, help='path to baseline image')
ap.add_argument('-o', '--output', required=True, help='path to output image')
args = vars(ap.parse_args())

# load pre-trained model
print('[INFO] loading model...')
model = load_model(config.MODEL_PATH)

# load input image, grab dimensions of input image and crop image such that it tiles nicely
print('[INFO] generating image...')
image = cv2.imread(args['image'])
h, w = image.shape[:2]
w -= int(w%config.SCALE)
h -= int(h%config.SCALE)
image = image[:h, :w]

# resize input image using bicubic interpolation then write baseline image to disk
scaled = misc.imresize(image, config.SCALE, interp='bicubic')
cv2.imwrite(args['baseline'], scaled)

# allocate memory for output image
output = np.zeros(scaled.shape)
h, w = output.shape[:2] 

# slide a window from left-to-right and top-to-bottom
for y in range(0, h - config.INPUT_DIM + 1, config.LABEL_SIZE):
    for x in range(0, w - config.INPUT_DIM + 1, config.LABEL_SIZE):
        # crop ROI from our scaled image
        crop = scaled[y:y + config.INPUT_DIM, x:x + config.INPUT_DIM]

        # predict on crop and store it
        P = model.predict(np.expand_dims(crop, axis=0))
        P = P.reshape((config.LABEL_SIZE, config.LABEL_SIZE, 3))
        output[y + config.PAD: y + config.PAD + config.LABEL_SIZE, x + config.PAD: x + config.PAD + config.LABEL_SIZE] = P

# remove any of black borders in output image caused by padding, then clip any values that fall outside range [0, 255]        
output = output[config.PAD:h - (h%config.INPUT_DIM + config.PAD), config.PAD:w - (w%config.INPUT_DIM + config.PAD)]
output = np.clip(output, 0, 255).astype('uint8')

# write output to disk
cv2.imwrite(args['output'], output)