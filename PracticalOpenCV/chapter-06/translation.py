# USAGE
# python translation.py --image ../images/trex.png
import numpy as np
import argparse
import imutils
import cv2

# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

# Load the image and show it
image = cv2.imread(args['image'])
cv2.imshow('Original', image)

M = np.array([[1, 0, 25], [0, 1, 50]], dtype='float32')
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow('Shifted Right and Down', shifted)
cv2.waitKey(0)

M = np.array([[1, 0, -50], [0, 1, -90]], dtype='float32')
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow('Shifted Left and Up', shifted)
cv2.waitKey(0)

# Finally, let's use our helper function in imutils.py to shift the image down 100 pixels
shifted = imutils.translate(image, 0, 100)
cv2.imshow('Shifted Down', shifted)
cv2.waitKey(0)