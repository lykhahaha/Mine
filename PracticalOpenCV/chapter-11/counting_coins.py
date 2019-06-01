# USAGE
# python counting_coins.py --image ..\images\coins.png
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
cv2.imshow('Blurred', image)

edged = cv2.Canny(blurred, 30, 150)
cv2.imshow('Edges', edged)

# cnts is list of array, each array is contour
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f'Count {len(cnts)} coins in this image')

coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
cv2.imshow('Coins', coins)
cv2.waitKey(0)

for i, cnt in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(cnt)
    
    print(f'Coin #{i+1}')
    coin = image[y:y+h, x:x+w]
    cv2.imshow('Coin', coin)

    mask = np.zeros(image.shape[:2], dtype='uint8')
    (center_X, center_Y), radius = cv2.minEnclosingCircle(cnt)
    mask = cv2.circle(mask, (int(center_X), int(center_Y)), int(radius), 255, -1)
    mask = mask[y:y+h, x:x+w]
    cv2.imshow('Masked coin', cv2.bitwise_and(coin, coin, mask=mask))
    cv2.waitKey(0)