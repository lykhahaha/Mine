# Usage
# python getting_and_setting.py --image ../images/trex.png
import argparse
import cv2

# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

# load and show image
image = cv2.imread(args['image'])
cv2.imshow('Original', image)

# access (b, g, r) value
b, g, r = image[0, 0]
print(f'Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}')

# assign value to pixel (b, g, r)
image[0, 0] = (0, 0, 255)
b, g, r = image[0, 0]
print(f'Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}')

# image slicing
corner = image[:100, :100]
cv2.imshow('Corner', corner)

image[:100, :100] = (0, 255, 0)
cv2.imshow('Updated', image)
cv2.waitKey(0)