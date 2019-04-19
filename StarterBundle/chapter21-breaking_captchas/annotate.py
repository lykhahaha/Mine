# USAGE
# python annotate.py --input downloads --annot dataset
from imutils import paths
import argparse
import imutils
import cv2
import os

# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='path to input directory of images')
ap.add_argument('-a', '--annot', required=True, help='path to output directory of annotations')
args = vars(ap.parse_args())

# grab image paths then initialize dictionary of character counts
image_paths = list(paths.list_images(args['input']))
counts = {}

# loop over the image paths
for i, image_path in enumerate(image_paths):
    # display an update to user
    print(f'[INFO] processing image {i+1}/{len(image_paths)}...')

    try:
        # load image and convert it to grayscale, then pad image to ensure digits caught on border of image are retained
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # Otsu threshold image to reveal digits
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # find contours in image, keep only four largest ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        # loop over contours
        for c in cnts:
            # compute bounding box for contour then extract digit
            x, y, w, h = cv2.boundingRect(c)
            roi = gray[y-5:y+h+5, x-5:x+w+5]

            # display character, make it large enough to see then wait for keypress
            cv2.imshow('ROI', imutils.resize(roi, width=28))
            key = cv2.waitKey(0)
            
            # if '`' is pressed, ignore character
            if key == ord("`"):
                print('[INFO] ignoring character')
                continue
            
            # grab key that was pressed and construct path
            key = chr(key).upper()
            dir_path = os.path.sep.join([args['annot'], key])

            # if output directory does not exist, create it
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # write labeled character to file
            count = counts.get(key, 1)
            p = os.path.sep.join([dir_path, f'{str(count).zfill(6)}.png'])
            cv2.imwrite(p, roi)

            # increase count for current key
            counts[key] = count + 1
    
    # we are trying to control-c out of the script, so break from the loop
    except KeyboardInterrupt:
        print('[INFO] manually leaving script')
        break
    # if any error occurs
    except:
        print('[INFO] skipping image...')