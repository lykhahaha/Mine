# USAGE
# python splitting_and_merging.py --image ../images/wave.png
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(B, G, R) = cv2.split(image)

# Show each channel individually
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)

# Merge the image back together again
merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)
cv2.waitKey(0)

zeros = np.zeros(image.shape[:2], dtype='uint8')
cv2.imshow('Red', cv2.merge([zeros, zeros, R]))
cv2.imshow('Green', cv2.merge([zeros, G, zeros]))
cv2.imshow('Blue', cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)