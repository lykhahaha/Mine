# USAGE
# python simple_detection.py --image beagle.png --confidence 0.8
from keras.applications import ResNet50, imagenet_utils
from pyimagesearch.utils.simple_obj_det import sliding_window, image_pyramid, classify_batch
from imutils.object_detection import non_max_suppression
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import time
import cv2

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='minimum probability to filter weak detections')
args = vars(ap.parse_args())

# initialize variables used for object detection procedure
INPUT_SIZE = (350, 350)
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (224, 224)
BATCH_SIZE = 64

# load our network weights from disk
print('[INFO] loading network...')
model = ResNet50(weights='imagenet')

# initialize object detection dictionary which maps class labels to their predicted bounding boxes and associated probability
labels = {}

# load input image from disk and grab its dimension
orig = cv2.imread(args['image'])
h, w = orig.shape[:2]

# resize input image to be a square
resized = cv2.resize(orig, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

# initialize batch ROIs and (x, y)-coords
batch_ROIs = []
batch_locs = []

# start timer
print('[INFO] detecting objects...')
start = time.time()

# loop over image pyramid
for image in image_pyramid(resized, scale=PYR_SCALE, min_size=ROI_SIZE):
    # loop over sliding window locations
    for x, y, roi in sliding_window(image, WIN_STEP, ROI_SIZE):
        # take ROI and pre-process it so we can classify region with Keras
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        roi = imagenet_utils.preprocess_input(roi)

        batch_ROIs.append(roi)
        batch_locs.append((x, y))
    
    if len(batch_ROIs) == BATCH_SIZE:
        batch_ROIs = np.vstack(batch_ROIs)
        # classify batch, then reset batch ROIs and (x, y)-coords
        labels = classify_batch(model, batch_ROIs, batch_locs, labels, min_prob=args['confidence'])

        # reset batch_ROIs and batch_locs
        batch_ROIs = []
        batch_locs = []

# check to see if there are any remaining ROIs that still need to be classified
if len(batch_ROIs) > 0:
    batch_ROIs = np.vstack(batch_ROIs)
    labels = classify_batch(model, batch_ROIs, batch_locs, labels, min_prob=args['confidence'])

# show how long detection preocess took
end = time.time()
print(f'[INFO] detections took {end-start:.4f} seconds')

# loop over labels for each of detected objects in image
count = 1
for k in labels.keys():
    # copy resized image for showing
    clone = resized.copy()

    # loop over all bounding boxes for label and draw them on image
    for box, prob in labels[k]:
        x1, y1, x2, y2 = box
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # show image without apply nms
        cv2.imwrite(f'without-nms-{count}.jpg', clone)

    # copy resized image for showing
    clone = resized.copy()
    # grab bounding boxes and prob for each detection, then apply nms to suppress weaker, overlapping detections
    boxes = np.array([p[0] for p in labels[k]])
    probs = np.array([p[1] for p in labels[k]])
    boxes = non_max_suppression(boxes, probs=probs)

    # loop over all bounding boxes for label that were not suppressed
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # show image without apply nms
        cv2.imwrite(f'with-nms-{count}.jpg', clone)
    count += 1
    #cv2.waitKey(0)