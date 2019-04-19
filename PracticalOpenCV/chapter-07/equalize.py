import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.style.use('ggplot')
plt.figure()
plt.plot(hist)
plt.title('Grayscale Histogram w/out equalizing')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.xlim([0, 256])

plt.savefig('Grayscale_histogram_w_out_equalizing.png')

eq = cv2.equalizeHist(image)
hist = cv2.calcHist([eq], [0], None, [256], [0, 256])

plt.style.use('ggplot')
plt.figure()
plt.plot(hist)
plt.title('Grayscale Histogram with equalizing')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.xlim([0, 256])

plt.savefig('Grayscale_histogram_with_equalizing.png')

cv2.imshow('Histogram equalization', np.hstack([image, eq]))
cv2.waitKey(0)