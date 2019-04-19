import cv2
import numpy as np

def padding(image, width=640, height=480, depth=3):
    image_padded = np.zeros((height, width, depth), dtype='uint8')
    if depth == 1:
        image_padded = np.zeros((height, width), dtype='uint8')

    