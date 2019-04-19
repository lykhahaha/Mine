# USAGE
# python flipping.py --image ../images/trex.png
import argparse
import cv2

# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

flipped = cv2.flip(image, 1)
cv2.imshow('Flipped Horizontally', flipped)

flipped = cv2.flip(image, 0)
cv2.imshow('Flipped Vertically', flipped)

flipped = cv2.flip(image, -1)
cv2.imshow('Flipped Horizontally & Vertically', flipped)
cv2.waitKey(0)