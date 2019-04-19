# USAGE
# python rotate.py --image ../images/trex.png
import numpy as np
import argparse
import cv2
import imutils

# Construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

# Grab the dimensions of the image and calculate the center of the image
h, w = image.shape[:2]
center = (w//2, h//2)

# Rotate our image by 45 degrees
M = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow('Rotated by 45 degrees', rotated)

# Rotate our image by -90 degrees
M = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow('Rotated by -90 degrees', rotated)
cv2.waitKey(0)

# Finally, let's use our helper function in imutils.py to rotate the image by 180 degrees (flipping it upside down)
rotated = imutils.rotate(image, 180)
cv2.imshow('Rotated by 180 degrees', rotated)
cv2.waitKey(0)