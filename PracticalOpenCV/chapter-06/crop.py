# USAGE
# python crop.py --image ../images/trex.png
import numpy as np
import argparse
import cv2

# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

cropped = image[30:120, 240:355]
cv2.imshow('T-Rex Face', cropped)
cv2.waitKey(0)