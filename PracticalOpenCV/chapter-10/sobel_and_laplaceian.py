# USAGE
# python sobel_and_laplaceian.py --image ..\images\coins.png
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', image)

lap = cv2.Laplacian(image, cv2.CV_64F)
lap = abs(lap).astype('uint8')
cv2.imshow('Laplacian', lap)
cv2.waitKey(0)

sobel_X = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobel_Y = cv2.Sobel(image, cv2.CV_64F, 0, 1)

sobel_X = abs(sobel_X).astype('uint8')
sobel_Y = abs(sobel_Y).astype('uint8')
sobel_combined = cv2.bitwise_or(sobel_X, sobel_Y)
cv2.imshow('Sobel X', sobel_X)
cv2.imshow('Sobel Y', sobel_Y)
cv2.imshow('Sobel Combined', sobel_combined)
cv2.waitKey(0)