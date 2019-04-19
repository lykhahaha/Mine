# USAGE
# python color_histograms.py --image ../images/beach.png
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow("Original", image)

chans = cv2.split(image)
colors = ('b', 'g', 'r')

plt.style.use('ggplot')

plt.figure()
plt.title("Each channel Histogram")
plt.xlabel('Bins')
plt.ylabel('# of pixels')

for chan, color in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()


fig, axes = plt.subplots(1, 3)
fig.tight_layout()
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = axes[0].imshow(hist, interpolation='nearest')
axes[0].set_title('2D Color Histogram for G and B')
fig.colorbar(p, ax=axes[0])

hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = axes[1].imshow(hist, interpolation='nearest')
axes[1].set_title('2D Color Histogram for G and R')
fig.colorbar(p, ax=axes[1])

ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = axes[2].imshow(hist, interpolation='nearest')
axes[2].set_title('2D Color Histogram for B and R')
fig.colorbar(p, ax=axes[2])
plt.show()

print(f'2D histogram shape: {hist.shape}, with {hist.flatten().shape[0]} values')