import cv2
import numpy as np

class CropPreprocessor:
    def __init__(self, height, width, horiz_flip=False, inter=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.inter = inter
        self.horiz_flip = horiz_flip

    def preprocess(self, image):
        h, w = image.shape[:2]
        d_h, d_w = (h-self.height)//2, (w-self.width)//2

        coords = [
            (0, 0, self.height, self.width),
            (0, w-self.width, self.height, w),
            (h-self.height, 0, h, self.width),
            (h-self.height, w-self.width, h, w),
            (d_h, d_w, h-d_h, w-d_w)
        ]

        crops = []
        
        for start_y, start_x, end_y, end_x in coords:
            crop = image[start_y:end_y, start_x:end_x]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)
        
        if self.horiz_flip:
            flips = [cv2.flip(crop, 1) for crop in crops]
            crops.extend(flips)

        return np.array(crops)