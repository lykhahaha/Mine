import numpy as np
import cv2

def translate(image, x, y):
    # Define the translation matrix and perform the translation
    M = np.array([[1, 0, x], [0, 1, y]], dtype='float32')
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def rotate(image, angle, center=None, scale=1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # If the center is None, initialize it as the center of the image
    if center is None:
        center = (w//2, h//2)
    
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    h, w = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        r = float(height)/h
        dim = (int(w*r), height)
    
    # otherwise, the height is None
    if height is None:
        r = float(width)/w
        dim = (width, int(h*r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized