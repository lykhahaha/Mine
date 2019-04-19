import imutils
import cv2
import numpy as np
"""
Only applied when dimensions of target (self.height, self.width) < dimensions of origin (image.shape)
"""

class AspectAwarePreprocessor:
    def __init__(self, width, height, depth=3, inter=cv2.INTER_AREA):
        self.width = width # 256
        self.height = height # 256
        self.depth = depth
        self.inter = inter
    
    def preprocess(self, image):
        # grab the dimension of image, then initialize deltas to use when cropping
        h, w = image.shape[:2] # h = 500, w = 557
        dW, dH = 0, 0
        h_ratio, w_ratio = h/self.height, w/self.width
        image_resized = np.zeros((self.height, self.width, self.depth), dtype='uint8')
        if self.depth == 1:
            image_resized = np.zeros((self.height, self.width), dtype='uint8')

        if self.height <= h and self.width <= w:
            # if width is smaller than height, resize along width and then update deltas to crop height to desired dimension
            if w_ratio < h_ratio:
                image = imutils.resize(image, width=self.width, inter=self.inter)
                dH = int((image.shape[0] - self.height)/2.0) # 310-256/2=27
            else:
                image = imutils.resize(image, height=self.height, inter=self.inter) # (256, 285)
                dW = int((image.shape[1] - self.width)/2.0) # (285-256)/2=13
            image = image[dH:h-dH, dW:w-dW]# (256, 257)

            image_resized = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        elif self.height >= h and self.width >= w:
            # if width is smaller than height, resize along width and then update deltas to crop height to desired dimension
            if w_ratio < h_ratio:
                image = imutils.resize(image, height=self.height, inter=self.inter)
                dW = int((self.width - image.shape[1])/2.0) # 310-256/2=27
                image_resized[:, dW:self.width-dW] = cv2.resize(image, (self.width-(2*dW), self.height), interpolation=self.inter)
            else:
                image = imutils.resize(image, width=self.width, inter=self.inter) # (256, 285)
                dH = int((self.height - image.shape[0])/2.0) # (285-256)/2=13
                image_resized[dH:h-dH, :] = cv2.resize(image, (self.width, self.height-(2*dH)), interpolation=self.inter)
        else:
            raise ValueError('Width and Height of resized image must be greater or less than those of origin image')

        return image_resized