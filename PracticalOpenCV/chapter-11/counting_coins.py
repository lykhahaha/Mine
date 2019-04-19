# USAGE
# python counting_coins.py --image ..\images\coins.png
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow('Blurred', image)

edged = cv2.Canny(blurred, 30, 150)
cv2.imshow('Edges', edged)

_, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f'Count {cnts} coins in this image')

coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
cv2.imshow('Coins', coins)
cv2.waitKey(0)