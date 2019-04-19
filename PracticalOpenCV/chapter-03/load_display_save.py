# USAGE
# python load_display_save.py --image trex.jpg
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
print(f'width: {image.shape[1]} pixels')
print(f'height: {image.shape[0]} pixels')
print(f'depth: {image.shape[2]} pixels')

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.imwrite('newimage.jpg', image)