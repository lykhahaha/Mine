# USAGE
# python otsu_and_riddler.py --image ..\images\coins.png
import numpy as np
import argparse
import mahotas
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('Image', image)

T = mahotas.thresholding.otsu(blurred)
print(f"Otsu's threshold: {T}")

thresh = image.copy()
thresh = np.where(thresh > T, 0, 255).astype('uint8')
#thresh = cv2.bitwise_not(thresh)
cv2.imshow('Otsu', thresh)

T = mahotas.thresholding.rc(blurred)
print(f'Riddler-Calvard: {T}')
thresh = image.copy()
thresh = np.where(thresh > T, 0, 255).astype('uint8')
cv2.imshow('Riddler-Calvard', thresh)

cv2.waitKey(0)