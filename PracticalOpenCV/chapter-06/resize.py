# USAGE
# python resize.py --image ../images/trex.png
import numpy as np
import imutils
import cv2
import argparse

# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

# ratio of the new image to the old image. Let's make our new image have a width of 150 pixels
r = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * r))

# Perform the actual resizing of the image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Resized (Width)', resized)

# Let's make the height of the resized image 50 pixels
r = 50.0 / image.shape[0]
dim = (int(image.shape[1]*r), 50)

resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Resized (Height)', resized)
cv2.waitKey(0)

# one line of resizing code
resized = imutils.resize(image, width = 100)
cv2.imshow('Resized via Function', resized)
cv2.waitKey(0)