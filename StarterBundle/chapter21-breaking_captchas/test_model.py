# USAGE
# python test_model.py --input downloads --model output/lenet.hdf5
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from pyimagesearch.utils.captchahelper import process
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='path to input directory of images')
ap.add_argument('-m', '--model', required=True, help='path to input model')
args = vars(ap.parse_args())

# load pretrained network
print('[INFO] loading pretrained network...')
model = load_model(args['model'])

# get random sample
image_paths = list(paths.list_images(args['input']))
image_paths = np.random.choice(image_paths, 10, replace=False)

# loop over image paths
for image_path in image_paths:
    # preprocess like part of annotate.py
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE) # padding

    # Otsu threshold to reveal digits
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find contours in image, keep only four largest ones
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    cnts = contours.sort_contours(cnts)[0]

    # initialize the output image as as grayscale image with 3 channels along with output predictions
    output = cv2.merge([gray]*3)
    predictions = []

    for c in cnts:
        # compute bounding box for contour then extract digit
        x, y, w, h = cv2.boundingRect(c)
        roi = gray[y-5:y+h+5, x-5:x+w+5]

        # pre-process the ROI and then classify it
        roi = process(roi, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(pred))

        #draw prediction on output image
        cv2.rectangle(output, (x-2, y-2), (x+w+4, y+h+4), (0, 255, 0), 1)
        cv2.putText(output, str(pred), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    print(f"[INFO] captcha: {''.join(predictions)}")
    cv2.imshow('Output', output)
    cv2.waitKey(0)