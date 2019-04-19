import numpy as np
import cv2

class CropPreprocessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        # store target image width, height, whether or not horizontal flips should be included, along with the interpolation method used when resizing
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image):
        # initialize the list of crops
        crops = []

        # grab the width and height of the image then use these dimensions to define the corners of the image based
        h, w = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height], # top-left
            [w - self.width, 0, w, self.height], # top-right
            [w - self.width, h - self.height, w, h], # bottom-right
            [0, h - self.height, self.width, h] # bottom-left
        ]

        # compute the center crop of the image as well
        dW = int(0.5*(w - self.width))
        dH = int(0.5*(h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        # loop over the coordinates, extract each of crops and resize each of them to fixed size
        for start_x, start_y, end_x, end_y in coords:
            crop = image[start_y:end_y, start_x: end_x]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)

        # check to see if horizontal flips should be taken
        if self.horiz:
            # compute horizontal mirror flips for each crop
            mirrors = [cv2.flip(image, 1) for image in crops]
            crops.extend(mirrors)

        return np.array(crops)